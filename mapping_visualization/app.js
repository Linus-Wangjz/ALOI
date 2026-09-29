(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var BATCH_SIZE = 4;
  var GB_WRITE_BITS_PER_CYCLE = 32;
  var DEVICE_COLORS = [
    { fill: "#dce7ff", strong: "#2058d6", text: "#173d93" },
    { fill: "#dff5f2", strong: "#0e9488", text: "#08675f" },
    { fill: "#fff0dc", strong: "#dc8524", text: "#81480d" },
    { fill: "#eee8ff", strong: "#7456d8", text: "#5038a1" },
    { fill: "#e1f1ff", strong: "#2581bf", text: "#15577f" },
    { fill: "#ffe5eb", strong: "#cf5572", text: "#91364d" },
    { fill: "#e9f3d9", strong: "#71953b", text: "#4c6924" },
    { fill: "#f2e9dd", strong: "#99714a", text: "#684928" }
  ];

  var state = {
    tp: 8,
    context: 32768,
    dram: "gddr6",
    precision: "bf16",
    mapping: "channel_group",
    level: 1,
    kernel: null,
    device: 0,
    animationToken: 0,
    toastTimer: null
  };

  var kernels = {
    q: {
      title: "Q projection · Wq",
      short: "Wq",
      kind: "dense",
      k: 8192,
      n: 8192,
      split: "column",
      globalLabel: "Global Wq [8192, 8192]",
      equation: "X [B, 8192] × Wq [8192, 8192] → Q [B, 64, 128]",
      subtitle: "The complete 8K × 8K matrix is partitioned along its output columns.",
      inputAxis: "K · input / reduction = 8192",
      outputAxis: "N · Q output = 64 heads × 128",
      localOutput: function (s) { return "Q [B, " + (64 / s.tp) + ", 128]"; },
      communication: "No TP collective; local Q heads flow directly into local attention."
    },
    kv: {
      title: "Fused K / V projections",
      short: "[Wk | Wv]",
      kind: "dense",
      k: 8192,
      n: 2048,
      split: "column",
      globalLabel: "Global fused [Wk | Wv] [8192, 2048]",
      equation: "X [B, 8192] × [Wk | Wv] [8192, 2048] → [K | V]",
      subtitle: "K and V columns for the same KV heads stay together on one TP device.",
      inputAxis: "K · input / reduction = 8192",
      outputAxis: "N · K 1024 + V 1024",
      localOutput: function (s) { return "K/V [B, " + (8 / s.tp) + ", 128] each"; },
      communication: "No TP collective; each device appends its own complete KV heads."
    },
    qk: {
      title: "Q × Kᵀ attention score",
      short: "QK",
      kind: "qk",
      split: "head-row",
      globalLabel: "Global K cache [8 KV heads, L, 128]",
      equation: "Q [B, 64, 128] × Kᵀ [B, 8, 128, L] → Score [B, 64, L]",
      subtitle: "The global cache is partitioned by KV head; every device keeps the full context of its heads.",
      inputAxis: "KV head × complete context L",
      outputAxis: "head dimension = 128",
      localOutput: function (s) { return "Score [B, " + (64 / s.tp) + ", " + contextLabel(s.context) + "]"; },
      communication: "No cross-device traffic; eight query heads reuse each local KV head."
    },
    sv: {
      title: "Score × V context aggregation",
      short: "SV",
      kind: "sv",
      split: "head-row",
      globalLabel: "Global V cache [8 KV heads, L, 128]",
      equation: "Score [B, 64, L] × V [B, 8, L, 128] → O [B, 64, 128]",
      subtitle: "V-cache ownership follows KV heads; context groups are reduced only inside the device.",
      inputAxis: "KV head × complete context L",
      outputAxis: "head dimension = 128",
      localOutput: function (s) { return "O [B, " + (64 / s.tp) + ", 128]"; },
      communication: "Context-group partials use device-local PNM reduction, not TP all-reduce."
    },
    wo: {
      title: "Output projection · Wo",
      short: "Wo",
      kind: "dense",
      k: 8192,
      n: 8192,
      split: "row",
      globalLabel: "Global Wo [8192, 8192]",
      equation: "O [B, 8192] × Wo [8192, 8192] → Y [B, 8192]",
      subtitle: "The same 8K × 8K shape as Wq, but partitioned along the input/reduction dimension.",
      inputAxis: "K · attention output = 8192",
      outputAxis: "N · hidden output = 8192",
      localOutput: function () { return "Partial Y [B, 8192]"; },
      communication: "Every device produces a full hidden-vector partial; TP all-reduce combines them."
    },
    w13: {
      title: "Fused gate / up projections",
      short: "[W1 | W3]",
      kind: "dense",
      k: 8192,
      n: 57344,
      split: "column",
      globalLabel: "Global fused [W1 | W3] [8192, 57344]",
      equation: "X [B, 8192] × [W1 | W3] [8192, 57344] → gate/up",
      subtitle: "The fused FFN output columns are partitioned across TP devices.",
      inputAxis: "K · hidden input = 8192",
      outputAxis: "N · gate 28672 + up 28672",
      localOutput: function (s) { return "gate/up [B, " + number(57344 / s.tp) + "]"; },
      communication: "No collective between W1/W3 and W2; the local FFN shard remains on-device."
    },
    w2: {
      title: "Down projection · W2",
      short: "W2",
      kind: "dense",
      k: 28672,
      n: 8192,
      split: "row",
      globalLabel: "Global W2 [28672, 8192]",
      equation: "FFN [B, 28672] × W2 [28672, 8192] → Y [B, 8192]",
      subtitle: "The FFN reduction dimension is partitioned across TP devices.",
      inputAxis: "K · FFN intermediate = 28672",
      outputAxis: "N · hidden output = 8192",
      localOutput: function () { return "Partial Y [B, 8192]"; },
      communication: "Every device produces a full hidden-vector partial; TP all-reduce combines them."
    },
    rms1: localKernel("Attention RMSNorm", "RMSNorm and element-wise scaling execute locally on the PNM vector path."),
    rope: localKernel("RoPE & KV-cache append", "RoPE executes locally, then each TP device writes only its owned KV heads."),
    softmax: localKernel("Streaming Softmax", "Online Softmax state stays on the local accelerator and feeds SV without a cross-device score workspace."),
    residual1: localKernel("Attention residual add", "The residual addition executes after the Wo TP all-reduce, with a replicated hidden vector."),
    rms2: localKernel("FFN RMSNorm", "RMSNorm executes locally on the replicated post-attention hidden vector."),
    silu: localKernel("SiLU(W1) × W3", "The fused activation and element-wise multiply consume the device-local FFN shard."),
    residual2: localKernel("FFN residual add", "The residual addition executes after the W2 TP all-reduce.")
  };

  function localKernel(title, subtitle) {
    return {
      title: title,
      short: title,
      kind: "local",
      equation: "Device-local PNM / vector operation",
      subtitle: subtitle,
      localOutput: function () { return "Replicated hidden state [B, 8192]"; },
      communication: "No weight-matrix partition is required for this operation."
    };
  }

  function number(value) {
    return Number(value).toLocaleString("en-US");
  }

  function contextLabel(value) {
    return value === 131072 ? "128K" : "32K";
  }

  function dramLabel(value) {
    return value === "lpddr4x" ? "LPDDR4X" : "GDDR6";
  }

  function bankCount() {
    return state.dram === "lpddr4x" ? 8 : 16;
  }

  function precisionProfile() {
    if (state.precision === "fp8") {
      return {
        key: "fp8",
        label: "FP8",
        elementBits: 8,
        burstLength: 32,
        dramColumns: 2048,
        systolicWidth: 32,
        halfWidth: 16
      };
    }
    return {
      key: "bf16",
      label: "BF16",
      elementBits: 16,
      burstLength: 16,
      dramColumns: 1024,
      systolicWidth: 16,
      halfWidth: 8
    };
  }

  function saLabel() {
    return "SA 4×" + precisionProfile().systolicWidth;
  }

  function mappingLabel() {
    return state.mapping === "reduction_split" ? "Reduction-Split Mapping" : "Channel Group Mapping";
  }

  function batched(text) {
    return String(text).replace(/\[B,/g, "[" + BATCH_SIZE + ",");
  }

  function svgEl(name, attrs, text) {
    var node = document.createElementNS(NS, name);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        if (attrs[key] !== null && attrs[key] !== undefined) {
          node.setAttribute(key, attrs[key]);
        }
      });
    }
    if (text !== undefined && text !== null) {
      node.textContent = text;
    }
    return node;
  }

  function append(parent, name, attrs, text) {
    var node = svgEl(name, attrs, text);
    parent.appendChild(node);
    return node;
  }

  function textLines(parent, x, y, lines, className, lineHeight, anchor) {
    var text = append(parent, "text", {
      x: x,
      y: y,
      "class": className || "svg-label",
      "text-anchor": anchor || "start"
    });
    lines.forEach(function (line, index) {
      append(text, "tspan", { x: x, dy: index === 0 ? 0 : (lineHeight || 14) }, line);
    });
    return text;
  }

  function fitRect(k, n, bounds) {
    var ratio = n / Math.max(k, 1);
    ratio = Math.max(0.55, Math.min(4, ratio));
    var width;
    var height;
    if (ratio >= 1) {
      width = bounds.w;
      height = Math.max(bounds.minH || 85, Math.min(bounds.h, width / ratio));
    } else {
      height = bounds.h;
      width = Math.max(bounds.minW || 130, Math.min(bounds.w, height * ratio));
    }
    return {
      x: bounds.x + (bounds.w - width) / 2,
      y: bounds.y + (bounds.h - height) / 2,
      w: width,
      h: height
    };
  }

  function globalShape(kernel) {
    if (kernel.kind === "qk" || kernel.kind === "sv") {
      return { k: state.context * 8, n: 128 };
    }
    return { k: kernel.k, n: kernel.n };
  }

  function localShape(kernel) {
    var shape = globalShape(kernel);
    if (kernel.kind === "qk" || kernel.kind === "sv") {
      return { k: state.context * (8 / state.tp), n: 128 };
    }
    if (kernel.split === "column") {
      return { k: shape.k, n: shape.n / state.tp };
    }
    return { k: shape.k / state.tp, n: shape.n };
  }

  function localShapeLabel(kernel) {
    if (kernel.kind === "qk" || kernel.kind === "sv") {
      return "[" + (8 / state.tp) + " KV heads, " + contextLabel(state.context) + ", 128]";
    }
    var shape = localShape(kernel);
    return "[" + number(shape.k) + ", " + number(shape.n) + "]";
  }

  function kvDepthSpec(kernel) {
    if (kernel.short === "Wq") {
      return { label: "KV GROUP", count: 8, planeK: 8192, planeN: 1024, noun: "KV groups" };
    }
    if (kernel.short === "[Wk | Wv]") {
      return { label: "KV HEAD", count: 8, planeK: 8192, planeN: 256, noun: "KV heads" };
    }
    if (kernel.kind === "qk" || kernel.kind === "sv") {
      return { label: "KV HEAD", count: 8, planeK: state.context, planeN: 128, noun: "KV heads" };
    }
    if (kernel.short === "Wo") {
      return { label: "KV GROUP", count: 8, planeK: 1024, planeN: 8192, noun: "KV groups" };
    }
    return null;
  }

  function physicalMatrixShape(kernel) {
    if (kernel.kind === "qk") return { k: 128, n: state.context };
    if (kernel.kind === "sv") return { k: state.context, n: 128 };
    return localShape(kernel);
  }

  function gemvEquation(kernel) {
    var shape = physicalMatrixShape(kernel);
    if (kernel.kind === "qk") {
      return "4 independent batches · per KV head & batch: Q_GQA [8 × 128] = 2 waves [4 × 128] × K_headᵀ [128 × " + contextLabel(state.context) + "] → score [8 × " + contextLabel(state.context) + "]";
    }
    if (kernel.kind === "sv") {
      return "4 independent batches · per KV head & batch: score_GQA [8 × " + contextLabel(state.context) + "] = 2 waves [4 × " + contextLabel(state.context) + "] × V_head [" + contextLabel(state.context) + " × 128] → o_GQA [8 × 128]";
    }
    return "X [4 × " + number(shape.k) + "] × W_local [" + number(shape.k) + " rows × " + number(shape.n) + " columns] → Y_local [4 × " + number(shape.n) + "]";
  }

  function headRange(device, total) {
    var perDevice = total / state.tp;
    var first = device * perDevice;
    return first + "–" + (first + perDevice - 1);
  }

  function updateControls() {
    document.querySelectorAll(".segmented-control").forEach(function (control) {
      var key = control.getAttribute("data-control");
      control.querySelectorAll("button").forEach(function (button) {
        var raw = button.getAttribute("data-value");
        var active = String(state[key]) === raw;
        var incompatible = key === "dram" && raw === "gddr6" && state.precision === "fp8";
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
        button.disabled = incompatible;
        button.title = incompatible ? "FP8 mapping is validated for LPDDR4X only" : "";
      });
    });
    document.getElementById("footerGeometry").textContent = mappingLabel() + " · " + precisionProfile().label + " · " + saLabel();
  }

  function updateSummary() {
    var bpc = bankCount();
    var html = [
      ["TP group", state.tp + " device" + (state.tp > 1 ? "s" : "")],
      ["Context", contextLabel(state.context)],
      ["Batch", BATCH_SIZE + " active rows"],
      ["Precision", precisionProfile().label + " · 4×" + precisionProfile().systolicWidth],
      ["Mapping", state.mapping === "reduction_split" ? "Full N / split K" : "Channel groups"],
      ["Device", "32 ch × " + bpc + " banks"]
    ].map(function (item) {
      return '<div class="summary-chip"><span>' + item[0] + "</span><strong>" + item[1] + "</strong></div>";
    }).join("");
    document.getElementById("overviewSummary").innerHTML = html;
  }

  function updateBreadcrumb() {
    var parts = ["Transformer Block"];
    if (state.level >= 2 && state.kernel) {
      parts.push(kernels[state.kernel].short);
    }
    if (state.level >= 3) {
      parts.push("Device " + state.device);
      parts.push(state.mapping === "reduction_split" ? "Reduction split" : "Channel groups");
    }
    document.getElementById("breadcrumb").innerHTML = parts.map(function (part, index) {
      return '<span class="' + (index === parts.length - 1 ? "current" : "") + '">' + part + "</span>" +
        (index < parts.length - 1 ? '<span class="separator">/</span>' : "");
    }).join("");
    document.getElementById("backButton").hidden = state.level === 1;
    document.getElementById("exportButton").hidden = state.level === 1;
  }

  function setLevel(level) {
    state.level = level;
    [1, 2, 3].forEach(function (value) {
      var section = document.getElementById("level" + value);
      var active = value === level;
      section.hidden = !active;
      section.classList.toggle("is-active", active);
    });
    updateBreadcrumb();
    updateUrl();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function selectKernel(id) {
    state.kernel = id;
    state.device = 0;
    state.animationToken += 1;
    setLevel(2);
    renderLevel2();
  }

  function enterDevice(device) {
    var kernel = kernels[state.kernel];
    if (!kernel || kernel.kind === "local") {
      return;
    }
    state.device = device;
    state.animationToken += 1;
    setLevel(3);
    renderLevel3();
  }

  function goBack() {
    state.animationToken += 1;
    if (state.level === 3) {
      setLevel(2);
      renderLevel2();
    } else {
      setLevel(1);
    }
  }

  function updateUrl() {
    if (!window.history || !window.history.replaceState) return;
    var params = new URLSearchParams();
    params.set("tp", state.tp);
    params.set("context", state.context);
    params.set("dram", state.dram);
    params.set("precision", state.precision);
    params.set("mapping", state.mapping);
    if (state.kernel) params.set("kernel", state.kernel);
    if (state.level > 1) params.set("level", state.level);
    if (state.level === 3) params.set("device", state.device);
    try {
      window.history.replaceState(null, "", window.location.pathname + "?" + params.toString());
    } catch (error) {
      // Some browsers intentionally restrict history mutation for file:// URLs.
      // The explorer remains fully functional when opened directly from disk.
    }
  }

  function readUrl() {
    var params = new URLSearchParams(window.location.search);
    var tp = Number(params.get("tp"));
    var context = Number(params.get("context"));
    var dram = params.get("dram");
    var precision = params.get("precision");
    var mapping = params.get("mapping");
    var kernel = params.get("kernel");
    var level = Number(params.get("level"));
    var device = Number(params.get("device"));
    if ([1, 2, 4, 8].indexOf(tp) >= 0) state.tp = tp;
    if ([32768, 131072].indexOf(context) >= 0) state.context = context;
    if (["gddr6", "lpddr4x"].indexOf(dram) >= 0) state.dram = dram;
    if (["bf16", "fp8"].indexOf(precision) >= 0) state.precision = precision;
    if (["channel_group", "reduction_split"].indexOf(mapping) >= 0) state.mapping = mapping;
    if (state.precision === "fp8") state.dram = "lpddr4x";
    if (kernel && kernels[kernel]) state.kernel = kernel;
    if (state.kernel && (level === 2 || level === 3)) state.level = level;
    if (state.level === 3 && Number.isFinite(device) && device >= 0 && device < state.tp) state.device = device;
  }

  function renderAll() {
    updateControls();
    updateSummary();
    updateBreadcrumb();
    if (state.level === 2) renderLevel2();
    if (state.level === 3) renderLevel3();
    updateUrl();
  }

  function renderLevel2() {
    var kernel = kernels[state.kernel];
    if (!kernel) return;
    document.getElementById("level2-title").textContent = kernel.title;
    document.getElementById("level2-subtitle").textContent = kernel.subtitle;
    document.getElementById("partitionEquation").textContent = batched(kernel.equation);
    document.getElementById("mappingBadge").textContent = mappingLabel() + " · intra-device policy (TP ownership is unchanged)";
    document.getElementById("replayPartition").hidden = kernel.kind === "local";

    if (kernel.kind === "local") {
      renderLocalPartition(kernel);
    } else {
      renderMatrixPartition(kernel);
    }
    renderPartitionFacts(kernel);
  }

  function svgRoot(host, viewBox) {
    host.innerHTML = "";
    var svg = svgEl("svg", {
      viewBox: viewBox,
      role: "img",
      "aria-label": host.getAttribute("aria-label") || "Interactive mapping diagram"
    });
    host.appendChild(svg);
    return svg;
  }

  function addArrowDefs(svg) {
    var defs = append(svg, "defs");
    var marker = append(defs, "marker", {
      id: "arrowhead",
      viewBox: "0 0 10 10",
      refX: 8,
      refY: 5,
      markerWidth: 5,
      markerHeight: 5,
      orient: "auto-start-reverse"
    });
    append(marker, "path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#91a0b5" });
  }

  function devicePositions(tp) {
    var columns = tp <= 4 ? tp : 4;
    var rows = Math.ceil(tp / columns);
    var cardW = 132;
    var cardH = 144;
    var gapX = 15;
    var gapY = 34;
    var totalW = columns * cardW + (columns - 1) * gapX;
    var startX = 595 + (555 - totalW) / 2;
    var totalH = rows * cardH + (rows - 1) * gapY;
    var startY = 92 + (390 - totalH) / 2;
    var result = [];
    for (var index = 0; index < tp; index += 1) {
      result.push({
        x: startX + (index % columns) * (cardW + gapX),
        y: startY + Math.floor(index / columns) * (cardH + gapY),
        w: cardW,
        h: cardH
      });
    }
    return result;
  }

  function sourcePiece(frame, index, split, tp) {
    if (split === "column") {
      return { x: frame.x + index * frame.w / tp, y: frame.y, w: frame.w / tp, h: frame.h };
    }
    return { x: frame.x, y: frame.y + index * frame.h / tp, w: frame.w, h: frame.h / tp };
  }

  function drawTensorDepth(svg, frame, depth) {
    var dx = 44;
    var dy = -34;
    var top = [];
    for (var head = depth.count - 1; head >= 0; head -= 1) {
      var a0 = head / depth.count;
      var a1 = (head + 1) / depth.count;
      var owner = Math.min(state.tp - 1, Math.floor(head / (depth.count / state.tp)));
      var headColor = DEVICE_COLORS[owner];
      top.push(append(svg, "polygon", {
        points: [
          (frame.x + a0 * dx) + "," + (frame.y + a0 * dy),
          (frame.x + frame.w + a0 * dx) + "," + (frame.y + a0 * dy),
          (frame.x + frame.w + a1 * dx) + "," + (frame.y + a1 * dy),
          (frame.x + a1 * dx) + "," + (frame.y + a1 * dy)
        ].join(" "),
        fill: headColor.fill,
        stroke: headColor.strong,
        "stroke-width": 0.9,
        "class": "tensor-depth-slice",
        "data-kv-head": head,
        "data-owner": owner
      }));
    }
    append(svg, "polygon", {
      points: [
        (frame.x + frame.w) + "," + frame.y,
        (frame.x + frame.w + dx) + "," + (frame.y + dy),
        (frame.x + frame.w + dx) + "," + (frame.y + frame.h + dy),
        (frame.x + frame.w) + "," + (frame.y + frame.h)
      ].join(" "),
      fill: "#edf2f8",
      stroke: "#9eacbf",
      "stroke-width": 1
    });

    append(svg, "path", {
      d: "M " + (frame.x + 2) + " " + (frame.y - 13) + " L " + (frame.x + dx - 1) + " " + (frame.y + dy - 13),
      stroke: "#66758a",
      "stroke-width": 1.2,
      "marker-end": "url(#arrowhead)"
    });
    append(svg, "text", {
      x: frame.x + dx / 2 + 4,
      y: frame.y + dy / 2 - 18,
      "class": "svg-kicker",
      "text-anchor": "middle",
      transform: "rotate(-38 " + (frame.x + dx / 2 + 4) + " " + (frame.y + dy / 2 - 18) + ")"
    }, depth.label + " · 8");

    var railX = frame.x + frame.w + dx + 15;
    var railY = frame.y + 3;
    var railH = frame.h - 6;
    var sources = [];
    for (var device = 0; device < state.tp; device += 1) {
      var rail = {
        x: railX,
        y: railY + device * railH / state.tp,
        w: 24,
        h: railH / state.tp
      };
      sources.push(rail);
      append(svg, "rect", {
        x: rail.x,
        y: rail.y,
        width: rail.w,
        height: rail.h,
        rx: 2,
        fill: DEVICE_COLORS[device].fill,
        stroke: DEVICE_COLORS[device].strong,
        "class": "depth-owner-rail"
      });
      append(svg, "text", {
        x: rail.x + rail.w / 2,
        y: rail.y + rail.h / 2 + 3,
        fill: DEVICE_COLORS[device].text,
        "font-size": rail.h < 22 ? 6.5 : 8,
        "font-weight": 780,
        "text-anchor": "middle"
      }, "D" + device);
    }
    append(svg, "text", { x: railX + 12, y: railY + railH + 16, "class": "svg-small", "text-anchor": "middle" }, "TP owners");
    return sources;
  }

  function renderMatrixPartition(kernel) {
    var host = document.getElementById("partitionViz");
    host.setAttribute("aria-label", kernel.title + " global matrix partitioned across " + state.tp + " devices");
    var svg = svgRoot(host, "0 0 1200 560");
    addArrowDefs(svg);
    var shape = globalShape(kernel);
    var depth = kvDepthSpec(kernel);
    var local = localShape(kernel);
    var visibleShape = depth ? { k: depth.planeK, n: depth.planeN } : shape;
    var frame = fitRect(visibleShape.k, visibleShape.n, { x: 55, y: 138, w: depth ? 330 : 410, h: 245, minW: depth ? 205 : 145, minH: 92 });
    var positions = devicePositions(state.tp);

    append(svg, "text", { x: 55, y: 48, "class": "svg-kicker" }, depth ? "GLOBAL LOGICAL 3D TENSOR" : "GLOBAL LOGICAL MATRIX");
    append(svg, "text", { x: 55, y: 72, "class": "svg-title" }, kernel.globalLabel);
    append(svg, "text", { x: 55, y: 89, "class": "svg-subtitle" },
      depth ? "Logical reshape [" + depth.count + " " + depth.noun + ", " + number(depth.planeK) + ", " + number(depth.planeN) + "] · split the depth axis" :
        kernel.split === "column" ? "Column parallel · split output N" : "Row parallel · split reduction K");

    var depthSources = depth ? drawTensorDepth(svg, frame, depth) : null;

    append(svg, "rect", {
      x: frame.x,
      y: frame.y,
      width: frame.w,
      height: frame.h,
      rx: 7,
      "class": "matrix-frame"
    });
    if (depth) {
      for (var planeLine = 1; planeLine < 4; planeLine += 1) {
        append(svg, "line", {
          x1: frame.x + frame.w * planeLine / 4,
          y1: frame.y,
          x2: frame.x + frame.w * planeLine / 4,
          y2: frame.y + frame.h,
          "class": "matrix-grid-line"
        });
      }
      append(svg, "text", {
        x: frame.x + frame.w / 2,
        y: frame.y + frame.h / 2 + 4,
        "class": "svg-mono",
        "text-anchor": "middle"
      }, "one " + (depth.label === "KV HEAD" ? "KV-head" : "KV-group") + " plane");
    }

    append(svg, "text", {
      x: frame.x + frame.w / 2,
      y: frame.y + frame.h + 27,
      "class": "svg-small",
      "text-anchor": "middle"
    }, depth ? "per-plane N = " + number(depth.planeN) : kernel.outputAxis);
    append(svg, "text", {
      x: frame.x - 23,
      y: frame.y + frame.h / 2,
      "class": "svg-small",
      "text-anchor": "middle",
      transform: "rotate(-90 " + (frame.x - 23) + " " + (frame.y + frame.h / 2) + ")"
    }, depth ? "per-plane K = " + number(depth.planeK) : kernel.inputAxis);

    for (var i = 0; i < state.tp; i += 1) {
      var color = DEVICE_COLORS[i];
      var source = depth ? depthSources[i] : sourcePiece(frame, i, kernel.split === "column" ? "column" : "row", state.tp);
      append(svg, "rect", {
        x: source.x,
        y: source.y,
        width: source.w,
        height: source.h,
        fill: color.fill,
        stroke: color.strong,
        "class": "source-slice"
      });
      append(svg, "text", {
        x: source.x + source.w / 2,
        y: source.y + source.h / 2 + 3,
        fill: color.text,
        "font-size": Math.min(source.w, source.h) < 38 ? 7 : 10,
        "font-weight": 760,
        "text-anchor": "middle"
      }, "D" + i);

      var pos = positions[i];
      var target = fitRect(local.k, local.n, {
        x: pos.x + 16,
        y: pos.y + 46,
        w: pos.w - 32,
        h: 61,
        minW: 35,
        minH: 22
      });

      append(svg, "path", {
        d: "M " + (source.x + source.w / 2) + " " + (source.y + source.h / 2) +
          " C 505 " + (source.y + source.h / 2) + ", 535 " + (target.y + target.h / 2) + ", " +
          (target.x - 7) + " " + (target.y + target.h / 2),
        "class": "partition-arrow"
      });

      append(svg, "rect", {
        x: pos.x,
        y: pos.y,
        width: pos.w,
        height: pos.h,
        rx: 11,
        fill: "#fff",
        stroke: color.strong,
        "class": "device-card-svg device-outline"
      });
      append(svg, "rect", {
        x: pos.x,
        y: pos.y,
        width: pos.w,
        height: 29,
        rx: 11,
        fill: color.strong
      });
      append(svg, "rect", {
        x: pos.x,
        y: pos.y + 19,
        width: pos.w,
        height: 10,
        fill: color.strong
      });
      append(svg, "text", {
        x: pos.x + 11,
        y: pos.y + 19,
        fill: "#fff",
        "font-size": 10,
        "font-weight": 770
      }, "DEVICE " + i);
      append(svg, "text", {
        x: pos.x + pos.w / 2,
        y: pos.y + 123,
        "class": "svg-mono",
        "text-anchor": "middle"
      }, localShapeLabel(kernel));
      append(svg, "text", {
        x: pos.x + pos.w / 2,
        y: pos.y + 137,
        fill: color.text,
        "font-size": 8,
        "font-weight": 720,
        "text-anchor": "middle"
      }, deviceAnnotation(kernel, i));

      append(svg, "rect", {
        x: source.x,
        y: source.y,
        width: source.w,
        height: source.h,
        rx: 3,
        fill: color.fill,
        stroke: color.strong,
        "data-move": "true",
        "data-sx": source.x,
        "data-sy": source.y,
        "data-sw": source.w,
        "data-sh": source.h,
        "data-tx": target.x,
        "data-ty": target.y,
        "data-tw": target.w,
        "data-th": target.h,
        "class": "moving-slice"
      });

      var hit = append(svg, "rect", {
        x: pos.x,
        y: pos.y,
        width: pos.w,
        height: pos.h,
        rx: 11,
        tabindex: "0",
        role: "button",
        "aria-label": "Inspect Device " + i,
        "class": "device-hit"
      });
      (function (deviceIndex) {
        hit.addEventListener("click", function () { enterDevice(deviceIndex); });
        hit.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") enterDevice(deviceIndex);
        });
      })(i);
    }

    append(svg, "text", { x: 595, y: 48, "class": "svg-kicker" }, "TP DEVICE GROUP");
    append(svg, "text", { x: 595, y: 72, "class": "svg-title" }, "TP = " + state.tp);
    var note = textLines(svg, 600, 516, [
      kernel.split === "row" ? "Each rank computes the same output coordinates." : "Each rank owns different output coordinates.",
      "Click any device to inspect its local matrix placement."
    ], "svg-subtitle arrival-note", 15);
    note.setAttribute("id", "partitionArrival");

    if (kernel.split === "row") {
      var collective = append(svg, "g", { "class": "collective-line-svg", id: "collectiveResult" });
      append(collective, "path", { d: "M 790 485 H 1000", stroke: "#7456d8", "stroke-width": 1.8 });
      append(collective, "circle", { cx: 790, cy: 485, r: 4, fill: "#7456d8" });
      append(collective, "circle", { cx: 1000, cy: 485, r: 4, fill: "#7456d8" });
      append(collective, "rect", { x: 835, y: 472, width: 120, height: 26, rx: 13, fill: "#efeafe", stroke: "#9b87dd" });
      append(collective, "text", { x: 895, y: 489, fill: "#5a43ad", "font-size": 9, "font-weight": 760, "text-anchor": "middle" }, "TP ALL-REDUCE");
    }

    animatePartition();
  }

  function deviceAnnotation(kernel, device) {
    if (kernel.kind === "qk" || kernel.kind === "sv" || kernel.short === "[Wk | Wv]") {
      return "KV heads " + headRange(device, 8);
    }
    if (kernel.short === "Wq") {
      return "Q heads " + headRange(device, 64);
    }
    return kernel.split === "column" ? "output shard " + device : "reduction shard " + device;
  }

  function animatePartition() {
    state.animationToken += 1;
    var token = state.animationToken;
    var movers = Array.prototype.slice.call(document.querySelectorAll("#partitionViz [data-move]"));
    var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var duration = reduced ? 1 : 760;
    var stagger = reduced ? 0 : Math.min(55, 260 / Math.max(movers.length, 1));
    var start = performance.now() + (reduced ? 0 : 170);

    function frame(now) {
      if (token !== state.animationToken) return;
      var complete = true;
      movers.forEach(function (node, index) {
        var localStart = start + index * stagger;
        var t = Math.max(0, Math.min(1, (now - localStart) / duration));
        if (t < 1) complete = false;
        var eased = 1 - Math.pow(1 - t, 3);
        ["x", "y", "w", "h"].forEach(function (key) {
          var sourceKey = key === "w" ? "sw" : key === "h" ? "sh" : "s" + key;
          var targetKey = key === "w" ? "tw" : key === "h" ? "th" : "t" + key;
          var source = Number(node.getAttribute("data-" + sourceKey));
          var target = Number(node.getAttribute("data-" + targetKey));
          var attr = key === "w" ? "width" : key === "h" ? "height" : key;
          node.setAttribute(attr, source + (target - source) * eased);
        });
      });
      if (!complete) {
        requestAnimationFrame(frame);
      } else {
        document.querySelectorAll("#partitionViz .arrival-note, #partitionViz .collective-line-svg").forEach(function (node) {
          node.classList.add("is-visible");
        });
      }
    }
    requestAnimationFrame(frame);
  }

  function renderLocalPartition(kernel) {
    var host = document.getElementById("partitionViz");
    var svg = svgRoot(host, "0 0 1200 560");
    addArrowDefs(svg);
    append(svg, "text", { x: 60, y: 56, "class": "svg-kicker" }, "DEVICE-LOCAL OPERATION");
    append(svg, "text", { x: 60, y: 82, "class": "svg-title" }, kernel.title);
    append(svg, "rect", { x: 65, y: 190, width: 340, height: 135, rx: 14, fill: "#f5f8fc", stroke: "#90a0b6" });
    append(svg, "text", { x: 235, y: 242, "class": "svg-title", "text-anchor": "middle" }, "Replicated activation");
    append(svg, "text", { x: 235, y: 269, "class": "svg-mono", "text-anchor": "middle" }, "[4, 8192]");
    append(svg, "text", { x: 235, y: 294, "class": "svg-subtitle", "text-anchor": "middle" }, "No persistent weight matrix");

    var positions = devicePositions(state.tp);
    positions.forEach(function (pos, index) {
      var color = DEVICE_COLORS[index];
      append(svg, "path", {
        d: "M 405 257 C 480 257, 515 " + (pos.y + pos.h / 2) + ", " + pos.x + " " + (pos.y + pos.h / 2),
        "class": "partition-arrow"
      });
      append(svg, "rect", { x: pos.x, y: pos.y, width: pos.w, height: pos.h, rx: 11, fill: "#fff", stroke: color.strong, "class": "device-card-svg" });
      append(svg, "rect", { x: pos.x, y: pos.y, width: pos.w, height: 29, rx: 11, fill: color.strong });
      append(svg, "rect", { x: pos.x, y: pos.y + 19, width: pos.w, height: 10, fill: color.strong });
      append(svg, "text", { x: pos.x + 11, y: pos.y + 19, fill: "#fff", "font-size": 10, "font-weight": 770 }, "DEVICE " + index);
      append(svg, "rect", { x: pos.x + 18, y: pos.y + 53, width: pos.w - 36, height: 38, rx: 6, fill: color.fill, stroke: color.strong });
      append(svg, "text", { x: pos.x + pos.w / 2, y: pos.y + 76, fill: color.text, "font-size": 9, "font-weight": 730, "text-anchor": "middle" }, "LOCAL PNM");
      append(svg, "text", { x: pos.x + pos.w / 2, y: pos.y + 118, "class": "svg-small", "text-anchor": "middle" }, "same hidden state");
    });
    textLines(svg, 60, 410, [
      kernel.subtitle,
      "Return to the Transformer Block to select a matrix or cache kernel for bank-level mapping."
    ], "svg-subtitle", 18);
  }

  function renderPartitionFacts(kernel) {
    var container = document.getElementById("partitionFacts");
    var facts;
    if (kernel.kind === "local") {
      facts = [
        ["TP behavior", "Replicated", "The vector state is present on every rank."],
        ["Persistent matrix", "None", "This operation does not partition a weight matrix."],
        ["Execution", "Device-local", kernel.subtitle],
        ["Collective", "None here", "Any preceding all-reduce is shown on its matrix kernel.", "local"]
      ];
    } else {
      var shape = globalShape(kernel);
      var local = localShape(kernel);
      var depth = kvDepthSpec(kernel);
      var splitText = depth ? depth.label + " / depth" : kernel.split === "column" ? "Output N / columns" : kernel.split === "row" ? "Reduction K / rows" : "KV-head groups";
      facts = [
        ["Global operand", depth ? "[8, " + number(depth.planeK) + ", " + number(depth.planeN) + "]" : "[" + number(shape.k) + ", " + number(shape.n) + "]", depth ? "Logical KV-aware reshape of " + kernel.globalLabel : kernel.globalLabel],
        ["TP partition axis", splitText, depth ? "Each device owns " + (8 / state.tp) + " contiguous " + depth.noun + "." : kernel.split === "column" ? "Vertical matrix slices" : "Horizontal matrix slices"],
        ["Per-device operand", localShapeLabel(kernel), batched(kernel.localOutput(state))],
        [kernel.split === "row" ? "Cross-device result" : "Next step", kernel.split === "row" ? "TP All-Reduce" : "Device-local", kernel.communication, kernel.split === "row" ? "collective" : "local"]
      ];
    }
    container.innerHTML = facts.map(factHtml).join("");
  }

  function factHtml(fact) {
    return '<article class="fact-card ' + (fact[3] || "") + '"><span>' + fact[0] + "</span><strong>" + fact[1] + "</strong><small>" + fact[2] + "</small></article>";
  }

  function physicalLayout(kernel) {
    var bpc = bankCount();
    var profile = precisionProfile();
    var totalBanks = 32 * bpc;
    var capacity = totalBanks * profile.burstLength;
    var local = localShape(kernel);
    var result = {
      banksPerChannel: bpc,
      precision: profile.label,
      elementBits: profile.elementBits,
      burstLength: profile.burstLength,
      dramColumns: profile.dramColumns,
      systolicWidth: profile.systolicWidth,
      halfWidth: profile.halfWidth,
      totalBanks: totalBanks,
      outputCapacity: capacity,
      localK: local.k,
      localN: local.n,
      outputTiles: 1,
      channelsPerGroup: 32,
      reductionGroups: 1,
      reductionSlice: local.k,
      mappingType: kernel.kind,
      activeChannels: 32,
      representativeChannels: [0]
    };

    if (kernel.kind === "dense") {
      if (state.mapping === "reduction_split") {
        result.channelsPerGroup = 1;
        result.reductionGroups = 32;
        result.reductionSlice = Math.ceil(local.k / 32);
        result.activeChannels = 32;
        result.representativeChannels = [0];
        return completeWaveLayout(kernel, result);
      }
      result.outputTiles = Math.ceil(local.n / capacity);
      result.activeChannels = result.outputTiles > 1
        ? 32
        : Math.ceil(local.n / (bpc * profile.burstLength));
      if (kernel.short === "Wq" || kernel.short === "[Wk | Wv]") {
        var tileOutputs = Math.min(local.n, capacity);
        result.channelsPerGroup = Math.ceil(tileOutputs / (bpc * profile.burstLength));
        result.reductionGroups = Math.floor(32 / result.channelsPerGroup);
        result.reductionSlice = Math.ceil(local.k / result.reductionGroups);
        result.activeChannels = result.channelsPerGroup * result.reductionGroups;
      }
      result.representativeChannels = [];
      for (var denseChannel = 0; denseChannel < Math.min(result.channelsPerGroup, result.activeChannels); denseChannel += 1) {
        result.representativeChannels.push(denseChannel);
      }
      return completeWaveLayout(kernel, result);
    }

    var localKvHeads = 8 / state.tp;
    result.localKvHeads = localKvHeads;
    result.queryWaves = 2;
    result.banksPerHead = totalBanks / localKvHeads;
    result.channelsPerHead = 32 / localKvHeads;
    if (kernel.kind === "qk") {
      result.stripedContextsPerChannel = Math.ceil(state.context / result.channelsPerHead);
      result.contextsPerChannel = state.mapping === "reduction_split" ? state.context : result.stripedContextsPerChannel;
      result.contextsPerBank = Math.ceil(state.context / result.banksPerHead);
      result.representativeChannels = [0];
      if (state.mapping === "reduction_split") {
        result.channelsPerGroup = 1;
        result.reductionGroups = 32;
        result.reductionSlice = 4;
        result.activeChannels = 32;
      }
      return completeWaveLayout(kernel, result);
    }

    if (state.mapping === "reduction_split") {
      result.physicalGroupChannels = 1;
      result.svMode = "32-way context reduction split";
      result.contextGroups = 32;
      result.contextsPerHalf = Math.ceil(state.context / 32);
      result.channelsPerGroup = 1;
      result.reductionGroups = 32;
      result.reductionSlice = result.contextsPerHalf;
      result.activeChannels = 32;
      result.representativeChannels = [0];
      return completeWaveLayout(kernel, result);
    }

    var physicalGroupChannels = 128 / (bpc * profile.halfWidth);
    result.physicalGroupChannels = physicalGroupChannels;
    result.svMode = "one KV head per channel set";
    result.contextGroups = result.channelsPerHead / physicalGroupChannels;
    result.contextsPerHalf = Math.ceil(state.context / result.contextGroups);
    result.representativeChannels = [];
    for (var c = 0; c < physicalGroupChannels; c += 1) result.representativeChannels.push(c);
    return completeWaveLayout(kernel, result);
  }

  function completeWaveLayout(kernel, layout) {
    var shape = physicalMatrixShape(kernel);
    var reductionSplit = state.mapping === "reduction_split";
    layout.mappingPolicy = state.mapping;
    layout.requiresChannelReduction = reductionSplit;
    layout.channelWaveWidth = layout.banksPerChannel * layout.burstLength;

    if (reductionSplit) {
      layout.groupTileK = shape.k;
      layout.groupTileN = shape.n;
      layout.channelTileK = Math.ceil(shape.k / 32);
      layout.channelTileN = shape.n;
    } else if (kernel.kind === "qk") {
      layout.groupTileK = 128;
      layout.groupTileN = state.context;
      layout.channelTileK = 128;
      layout.channelTileN = layout.stripedContextsPerChannel;
    } else if (kernel.kind === "sv") {
      layout.groupTileK = layout.contextsPerHalf;
      layout.groupTileN = 128;
      layout.channelTileK = layout.contextsPerHalf;
      layout.channelTileN = Math.ceil(128 / Math.max(layout.physicalGroupChannels, 1));
    } else {
      layout.groupTileK = layout.reductionSlice;
      layout.groupTileN = Math.min(layout.localN, layout.outputCapacity);
      layout.channelTileK = layout.reductionSlice;
      layout.channelTileN = Math.min(layout.groupTileN, layout.channelWaveWidth);
    }

    layout.physicalSaPartitions = kernel.kind === "sv"
      ? Math.max(1, Math.floor(layout.systolicWidth / layout.halfWidth))
      : 1;
    layout.saPartitionWidth = kernel.kind === "sv" ? layout.halfWidth : layout.systolicWidth;
    layout.outputPartitionsPerHead = kernel.kind === "sv"
      ? Math.max(1, Math.ceil(layout.channelTileN / (layout.banksPerChannel * layout.saPartitionWidth)))
      : 1;
    layout.availableSvStreams = kernel.kind === "sv"
      ? Math.max(1, Math.floor(layout.physicalSaPartitions / layout.outputPartitionsPerHead))
      : 1;
    layout.parallelKParts = kernel.kind === "sv"
      ? layout.availableSvStreams
      : 1;
    layout.gbInputStreams = kernel.kind === "sv"
      ? layout.parallelKParts
      : 1;
    if (kernel.kind === "sv") {
      if (layout.parallelKParts > 1) {
        layout.svPartitionMode = "k_chunks";
        layout.svMode = "one KV head · two K streams";
      } else if (layout.outputPartitionsPerHead > 1) {
        layout.svPartitionMode = "output_slices";
        layout.svMode = "two N halves share one K stream";
      } else {
        layout.svPartitionMode = "single";
        layout.svMode = "one K stream";
      }
    }
    layout.kChunkCapacity = Math.floor(layout.dramColumns / (BATCH_SIZE * layout.gbInputStreams));
    if (kernel.kind === "qk") layout.kChunkCapacity = Math.min(128, layout.kChunkCapacity);
    layout.kCoveragePerWave = layout.kChunkCapacity * layout.parallelKParts;
    layout.gbKCoveragePerWave = layout.kChunkCapacity * layout.gbInputStreams;
    layout.waveK = Math.min(layout.channelTileK, layout.kChunkCapacity);
    layout.waveN = Math.min(layout.channelTileN, layout.channelWaveWidth);
    layout.outputWaves = Math.ceil(layout.channelTileN / Math.max(layout.channelWaveWidth, 1));
    layout.kChunks = Math.ceil(layout.channelTileK / Math.max(layout.kCoveragePerWave, 1));
    layout.totalWaves = layout.outputWaves * layout.kChunks;
    layout.wavesPerChannel = layout.totalWaves;
    layout.sequenceWaves = layout.totalWaves;
    return layout;
  }

  function cycleHeadSchedule(kernel) {
    if (kernel.kind !== "qk" && kernel.kind !== "sv") {
      return {
        factor: 1,
        kvHeadRounds: 1,
        batchRounds: 1,
        gqaWaves: 1,
        formula: "4 batches occupy the four SA rows together; head coordinates are already included in N",
        short: "1 Batch-4 launch"
      };
    }
    if (state.mapping === "reduction_split") {
      var localKvHeads = 8 / state.tp;
      return {
        factor: localKvHeads * 8,
        kvHeadRounds: localKvHeads,
        batchRounds: BATCH_SIZE,
        gqaWaves: 2,
        formula: localKvHeads + " KV-head round(s) × 4 batch rounds × 2 GQA waves (4 heads/wave)",
        short: localKvHeads + " KV × 4 B × 2 GQA"
      };
    }
    return {
      factor: 8,
      kvHeadRounds: 1,
      batchRounds: BATCH_SIZE,
      gqaWaves: 2,
      formula: "4 batch rounds × 2 GQA waves (4 heads/wave); local KV heads are spatial",
      short: "4 B × 2 GQA"
    };
  }

  function serialNPasses(kernel, layout) {
    if (kernel.kind === "dense" && state.mapping === "channel_group") {
      return layout.outputTiles * layout.outputWaves;
    }
    return layout.outputWaves;
  }

  function interChannelContributors(kernel, layout) {
    if (state.mapping === "reduction_split") return 32;
    if (kernel.kind === "qk") return 1;
    if (kernel.kind === "sv") return Math.max(1, layout.contextGroups);
    return Math.max(1, layout.reductionGroups);
  }

  function reductionSetsPerNPass(kernel, layout) {
    if (state.mapping === "reduction_split") return 1;
    if (kernel.kind === "sv") return Math.max(1, layout.physicalGroupChannels);
    if (kernel.kind === "dense") return Math.max(1, layout.channelsPerGroup);
    return 1;
  }

  function attentionScheduleTerms(head) {
    var terms = [];
    if (head.kvHeadRounds > 1) terms.push(head.kvHeadRounds + " KV-head rounds");
    terms.push(head.batchRounds + " batch rounds");
    terms.push(head.gqaWaves + " GQA waves");
    return terms;
  }

  function svPartitionSummary(layout) {
    var split = "SA 4×" + layout.systolicWidth + " → " + layout.physicalSaPartitions + " × 4×" + layout.saPartitionWidth;
    if (layout.svPartitionMode === "k_chunks") return split + " · two K streams/head";
    if (layout.svPartitionMode === "output_slices") return split + " · two N halves/K stream";
    return split;
  }

  function svPartitionExplanation(layout) {
    var prefix = "The 4×" + layout.systolicWidth + " array is partitioned into " + layout.physicalSaPartitions + " physical 4×" + layout.saPartitionWidth + " halves. ";
    if (layout.svPartitionMode === "k_chunks") {
      return prefix + "Both halves process the same four GQA heads for one KV head and consume distinct K chunks. Each fetched " + layout.saPartitionWidth + "-element V slice broadcasts across the four SA rows.";
    }
    if (layout.svPartitionMode === "output_slices") {
      return prefix + "The halves cover disjoint N slices and share one K stream. Each fetched " + layout.saPartitionWidth + "-element V slice broadcasts across the same four GQA rows.";
    }
    return prefix + "The selected head uses one K-stream path.";
  }

  function cycleMetricHtml(label, value, formula, note, tone) {
    return '<article class="cycle-metric ' + (tone || "") + '">' +
      '<span>' + label + '</span>' +
      '<strong>' + value + '</strong>' +
      '<code>' + formula + '</code>' +
      '<small>' + note + '</small>' +
      '</article>';
  }

  function renderCycleStats(kernel, layout) {
    var head = cycleHeadSchedule(kernel);
    var attention = kernel.kind === "qk" || kernel.kind === "sv";
    var nPasses = serialNPasses(kernel, layout);
    var aggregateKPerFill = Math.min(layout.channelTileK, layout.kCoveragePerWave);
    var fullFillCycles = Math.ceil(BATCH_SIZE * aggregateKPerFill * layout.elementBits / GB_WRITE_BITS_PER_CYCLE);
    var gbCyclesPerSchedule = Math.ceil(BATCH_SIZE * layout.channelTileK * layout.elementBits / GB_WRITE_BITS_PER_CYCLE);
    var totalGbCycles = gbCyclesPerSchedule * head.factor;
    var fillEvents = layout.kChunks * head.factor;
    var edgeFill = layout.channelTileK % layout.kCoveragePerWave !== 0;
    var gbFormula = number(fullFillCycles) + "/full fill × " + layout.kChunks + " K fill(s) × " + head.short;
    var gbRowNote = attention
      ? "One batch's four GQA heads at " + layout.precision + ". "
      : "Four batch rows at " + layout.precision + ". ";
    var svGbNote = layout.svPartitionMode === "k_chunks"
        ? layout.parallelKParts + " K-part streams × " + number(layout.waveK) + " values/part/row for one KV head. "
        : layout.svPartitionMode === "output_slices"
          ? "One " + number(layout.waveK) + "-value K stream is broadcast to both N halves. "
          : gbRowNote;
    var gbNote = (kernel.kind === "sv" ? svGbNote : gbRowNote) +
      number(nPasses) + " N pass(es) reuse GB; " + number(fillEvents) + " total fill event(s)" + (edgeFill ? ", last fill partial." : ".");

    var reductionCyclesPerNPass = Math.ceil(layout.channelTileK / layout.parallelKParts);
    var totalSaCycles = reductionCyclesPerNPass * nPasses * head.factor;
    var saFormula = kernel.kind === "sv"
      ? number(layout.channelTileK) + " K ÷ " + layout.parallelKParts + " K path(s) × " + number(nPasses) + " N pass(es) × " + head.short
      : number(reductionCyclesPerNPass) + " K/SA path × " + number(nPasses) + " N pass(es) × " + head.short;
    var saNote = (kernel.kind === "sv" ? svPartitionExplanation(layout) + " " : "") +
      layout.kChunks + " physical K wave(s) per N pass; SA startup and drain overhead are excluded.";
    if (kernel.kind === "sv") saNote += " At fixed TP, both mappings now have the same ideal SV SA cycles; GB traffic and inter-channel reductions may differ.";

    var contributors = interChannelContributors(kernel, layout);
    var reductionSets = reductionSetsPerNPass(kernel, layout);
    var reductionEvents = contributors > 1 ? nPasses * reductionSets * head.factor : 0;
    var pairwiseMerges = reductionEvents * Math.max(0, contributors - 1);
    var reductionTerms = attention ? attentionScheduleTerms(head) : [];
    reductionTerms.push(number(nPasses) + " N pass(es)");
    reductionTerms.push(reductionSets + " output set(s)/pass");
    reductionTerms.push("(" + contributors + "→1)");
    var reductionFormula = contributors > 1
      ? reductionTerms.join(" × ")
      : "N-only placement · no channel-spanning K split";
    var reductionUnitN = Math.min(layout.channelTileN, layout.banksPerChannel * layout.systolicWidth);
    var reductionUnitCapacityN = layout.banksPerChannel * layout.systolicWidth;
    var reductionRows = attention ? "four GQA heads from one batch" : "four batches";
    var reductionPayload = "[4 × " + number(reductionUnitN) + "] active" +
      (reductionUnitN < reductionUnitCapacityN ? " in [4 × " + number(reductionUnitCapacityN) + "] capacity" : "") +
      " (" + reductionRows + ")";
    var reductionNote = contributors > 1
      ? "One reduction-engine merge carries " + reductionPayload + ". K chunks accumulate inside each channel first, so there is no ×K-chunk factor."
      : "No inter-channel reduction is required for this mapping.";
    if (kernel.split === "row") reductionNote += " TP All-Reduce is separate and excluded.";

    var criticalPath = attention
      ? (state.mapping === "channel_group"
          ? "Critical path · channels and local KV heads run spatially; batches and GQA waves run in rounds"
          : "Critical path · channels run in lockstep; KV heads, batches and GQA waves run in rounds")
      : "Critical path · active channels run in parallel; four batches occupy the SA rows";

    var container = document.getElementById("cycleStats");
    container.setAttribute("data-gb-fill-cycles", totalGbCycles);
    container.setAttribute("data-sa-compute-cycles", totalSaCycles);
    container.setAttribute("data-inter-channel-merges", pairwiseMerges);
    container.setAttribute("data-reduction-events", reductionEvents);
    container.setAttribute("data-reduction-contributors", contributors);
    container.setAttribute("data-reduction-sets-per-n-pass", reductionSets);
    container.setAttribute("data-serial-n-passes", nPasses);
    container.setAttribute("data-serial-head-factor", head.factor);
    container.setAttribute("data-batch-rounds", head.batchRounds);
    container.setAttribute("data-gqa-waves", head.gqaWaves);
    container.setAttribute("data-kv-head-rounds", head.kvHeadRounds);
    container.innerHTML =
      '<div class="cycle-panel-heading">' +
        '<div><span>DEVICE CYCLE ESTIMATE</span><strong>' + criticalPath + '</strong></div>' +
        '<small>GB write = ' + GB_WRITE_BITS_PER_CYCLE + ' bits/cycle · ' + head.formula + '</small>' +
      '</div>' +
      '<div class="cycle-metric-grid">' +
        cycleMetricHtml("GB fill cycles", number(totalGbCycles) + " cycles", gbFormula, gbNote, "gb") +
        cycleMetricHtml("SA compute cycles", number(totalSaCycles) + " cycles", saFormula, saNote, "sa") +
        cycleMetricHtml("Inter-channel reductions", number(pairwiseMerges) + " merges", reductionFormula, reductionNote, "reduce") +
      '</div>';
  }

  function renderLevel3() {
    var kernel = kernels[state.kernel];
    if (!kernel || kernel.kind === "local") return;
    var layout = physicalLayout(kernel);
    document.getElementById("level3-title").textContent = kernel.short + " · Device " + state.device;
    document.getElementById("level3-subtitle").textContent = mappingLabel() + " · " +
      (kernel.kind === "qk" || kernel.kind === "sv"
        ? "show one batch's complete eight-head thin GEMM, then follow GQA H0–H3 through channels, GB and " + saLabel() + "."
        : "follow batch 4 through the TP-local [K rows × N columns] operand, channels, GB and " + saLabel() + ".");
    document.getElementById("physicalEquation").textContent = gemvEquation(kernel);
    document.getElementById("physicalHint").textContent = "Click the tile or one of the four phases";
    renderCycleStats(kernel, layout);
    renderPhysicalSvg(kernel, layout);
    renderPhysicalFacts(kernel, layout);
  }

  function physicalCanvasLayout(kernel, matrixShape) {
    var attention = kernel.kind === "qk" || kernel.kind === "sv";
    var matrixBounds = attention
      ? { x: 70, y: 350, w: 445, h: 200, minW: 135, minH: 105 }
      : { x: 70, y: 270, w: 445, h: 255, minW: 135, minH: 105 };
    var frame = fitRect(matrixShape.k, matrixShape.n, matrixBounds);
    var tileMetaY = frame.y + frame.h + 55;
    return {
      viewBox: "0 0 1360 1040",
      frame: frame,
      batch: { x: 62, y: 171, width: 82, height: 40, gap: 29, captionY: 239 },
      vector: { x: 70, y: attention ? 275 : 174, width: 445 },
      tileMetaY: tileMetaY,
      tileAction: { x: 366, y: tileMetaY - 12, width: 149, height: 36 },
      wavePanel: { x: 54, y: tileMetaY + 34, width: 481, height: 158 },
      ruleKickerY: tileMetaY + 218,
      pipelineKickerY: 930,
      phaseY: 950
    };
  }

  function renderPhysicalSvg(kernel, layout) {
    var host = document.getElementById("physicalViz");
    host.setAttribute("aria-label", kernel.title + " physical mapping inside Device " + state.device);
    var matrixShape = physicalMatrixShape(kernel);
    var canvas = physicalCanvasLayout(kernel, matrixShape);
    var frame = canvas.frame;
    var svg = svgRoot(host, canvas.viewBox);
    addArrowDefs(svg);
    var color = DEVICE_COLORS[state.device];

    append(svg, "text", { x: 54, y: 49, "class": "svg-kicker" }, "TP-LOCAL OPERAND");
    append(svg, "text", { x: 54, y: 75, "class": "svg-title" }, "Device " + state.device + " · " + localShapeLabel(kernel));
    append(svg, "text", { x: 54, y: 94, "class": "svg-subtitle" }, physicalOperandSubtitle(kernel, layout));

    append(svg, "text", { x: 54, y: 126, "class": "svg-kicker" }, kernel.kind === "qk" || kernel.kind === "sv" ? "BATCH-INDEPENDENT ATTENTION" : "BATCHED GEMM INPUT");
    append(svg, "text", { x: 54, y: 143, "class": "svg-subtitle" }, kernel.kind === "qk" || kernel.kind === "sv" ? "Four independent batch tensors; Batch 0 is expanded and the other three are faded" : "X [4 × K] is a thin matrix; each row feeds one SA row through GB");
    if (kernel.kind === "qk" || kernel.kind === "sv") drawBatchCubes(svg, kernel, layout, color, canvas);

    drawLocalMatrixGrid(svg, frame, kernel, layout, color);

    append(svg, "text", { x: frame.x + frame.w / 2, y: frame.y + frame.h + 27, "class": "svg-small", "text-anchor": "middle" }, "N · " + number(matrixShape.n) + " columns (output axis)");
    append(svg, "text", {
      x: frame.x - 23,
      y: frame.y + frame.h / 2,
      "class": "svg-small",
      "text-anchor": "middle",
      transform: "rotate(-90 " + (frame.x - 23) + " " + (frame.y + frame.h / 2) + ")"
    }, "K · " + number(matrixShape.k) + " rows (reduction axis)");

    var selected = selectedTile(frame, kernel, layout);
    drawSelectedTile(svg, selected, kernel, layout, color, canvas);
    drawWavePanel(svg, kernel, layout, color, canvas);

    append(svg, "path", {
      d: "M " + (selected.x + selected.w) + " " + (selected.y + selected.h / 2) +
        " C 535 " + (selected.y + selected.h / 2) + ", 555 245, 650 245",
      "class": "mapping-path",
      id: "mappingPath"
    });

    var bankPanel = append(svg, "g", { id: "bankPanel", "class": "bank-panel-svg" });
    append(bankPanel, "rect", { x: 590, y: 35, width: 720, height: 650, rx: 16, fill: "#fbfcfe", stroke: "#cdd6e3" });
    append(bankPanel, "text", { x: 615, y: 65, "class": "svg-kicker" }, "PHYSICAL DEVICE");
    append(bankPanel, "text", { x: 615, y: 88, "class": "svg-title" }, dramLabel(state.dram) + " · " + layout.precision + " · 32 channels × " + layout.banksPerChannel + " banks");
    append(bankPanel, "text", { x: 615, y: 105, "class": "svg-subtitle" }, bankPanelSubtitle(kernel, layout) + " · " + layout.burstLength + " values/256-bit bank word");

    var bankGeometry = drawBankGrid(bankPanel, kernel, layout, color);
    var runtimeGeometry = drawRuntimePanel(bankPanel, kernel, layout);
    drawGemvVector(svg, frame, kernel, layout, color, runtimeGeometry, canvas);
    drawPackets(svg, selected, bankGeometry, layout, kernel, color);
    drawExecutionPhases(svg, canvas.phaseY, kernel);
    wireChannelGroupHighlight(svg);

    append(svg, "text", { x: 54, y: canvas.ruleKickerY, "class": "svg-kicker" }, "SELECTED TILE RULE");
    textLines(svg, 54, canvas.ruleKickerY + 24, selectedRuleLines(kernel, layout), "svg-subtitle", 17);
    append(svg, "text", { x: 54, y: canvas.pipelineKickerY, "class": "svg-kicker" }, "INTERACTIVE PIPELINE · PHASE 4 IS ATTACHED TO THE C0 WAVE PANEL");

    svg.setAttribute("data-bank-count", layout.banksPerChannel);
    svg.setAttribute("data-precision", state.precision);
    svg.setAttribute("data-sa-width", layout.systolicWidth);
    svg.setAttribute("data-burst-length", layout.burstLength);
    svg.setAttribute("data-output-capacity", layout.outputCapacity);
    svg.setAttribute("data-reduction-groups", layout.reductionGroups);
    svg.setAttribute("data-reduction-slice", layout.reductionSlice);
    svg.setAttribute("data-channels-per-group", layout.channelsPerGroup);
    svg.setAttribute("data-mapping-policy", layout.mappingPolicy);
    svg.setAttribute("data-channel-tile-k", layout.channelTileK);
    svg.setAttribute("data-channel-tile-n", layout.channelTileN);
    svg.setAttribute("data-wave-k", layout.waveK);
    svg.setAttribute("data-wave-n", layout.waveN);
    svg.setAttribute("data-parallel-k-parts", layout.parallelKParts);
    svg.setAttribute("data-physical-sa-partitions", layout.physicalSaPartitions);
    svg.setAttribute("data-sa-partition-width", layout.saPartitionWidth);
    svg.setAttribute("data-sv-partition-mode", layout.svPartitionMode || "none");
    svg.setAttribute("data-gb-input-streams", layout.gbInputStreams);
    svg.setAttribute("data-k-coverage-per-wave", layout.kCoveragePerWave);
    svg.setAttribute("data-output-waves", layout.outputWaves);
    svg.setAttribute("data-k-chunks", layout.kChunks);
    svg.setAttribute("data-total-waves", layout.totalWaves);
    setPhase(1);
  }

  function physicalOperandSubtitle(kernel, layout) {
    if (kernel.kind === "qk") {
      return layout.localKvHeads + " local KV head(s) · representative head shown · complete " + contextLabel(state.context) + " context";
    }
    if (kernel.kind === "sv") {
      return layout.localKvHeads + " local KV head(s) · representative head shown · dual-half V-cache packing";
    }
    return kernel.split === "column" ? "TP-local output-column shard" : "TP-local reduction-row shard";
  }

  function drawBatchCubes(svg, kernel, layout, color, canvas) {
    var frame = canvas.frame;
    var startX = canvas.batch.x;
    var y = canvas.batch.y;
    var width = canvas.batch.width;
    var height = canvas.batch.height;
    var dx = 11;
    var dy = -8;
    for (var batch = 0; batch < BATCH_SIZE; batch += 1) {
      var x = startX + batch * (width + canvas.batch.gap);
      var active = batch === 0;
      var group = append(svg, "g", {
        "class": "batch-cube " + (active ? "is-current" : "is-muted"),
        "data-batch-cube": batch
      });
      append(group, "polygon", {
        points: x + "," + y + " " + (x + dx) + "," + (y + dy) + " " + (x + width + dx) + "," + (y + dy) + " " + (x + width) + "," + y,
        fill: active ? color.fill : "#edf1f6",
        stroke: active ? color.strong : "#aeb9c8"
      });
      append(group, "polygon", {
        points: (x + width) + "," + y + " " + (x + width + dx) + "," + (y + dy) + " " + (x + width + dx) + "," + (y + height + dy) + " " + (x + width) + "," + (y + height),
        fill: active ? color.fill : "#e3e8ef",
        stroke: active ? color.strong : "#aeb9c8"
      });
      append(group, "rect", {
        x: x,
        y: y,
        width: width,
        height: height,
        rx: 4,
        fill: active ? "#ffffff" : "#f4f6f9",
        stroke: active ? color.strong : "#aeb9c8",
        "stroke-width": active ? 1.8 : 0.9
      });
      append(group, "text", { x: x + width / 2, y: y + 17, "class": "svg-mono", "text-anchor": "middle" }, "BATCH " + batch);
      append(group, "text", { x: x + width / 2, y: y + 31, "class": "batch-cube-label", "text-anchor": "middle" }, (kernel.kind === "qk" ? "K" : "V") + " [" + layout.localKvHeads + ", L, 128]");
    }
    append(svg, "path", {
      d: "M " + (startX + width / 2) + " " + (y + height + 6) + " V " + (canvas.vector.y - 12),
      fill: "none",
      stroke: color.strong,
      "stroke-width": 1.3,
      "marker-end": "url(#arrowhead)"
    });
    append(svg, "text", { x: startX + 102, y: canvas.batch.captionY, fill: color.text, "font-size": 7.5, "font-weight": 760 }, "expand Batch 0 · B1–B3 use the identical mapping");
  }

  function drawLocalMatrixGrid(svg, frame, kernel, layout, color) {
    if (state.mapping === "reduction_split") {
      drawReductionSplitMatrixGrid(svg, frame, kernel, layout, color);
      return;
    }
    var groupCount = kernel.kind === "sv" ? Math.min(layout.contextGroups, 32) : kernel.kind === "qk" ? 1 : Math.min(layout.reductionGroups, 32);
    var channelsInGroup = kernel.kind === "qk" ? layout.channelsPerHead : kernel.kind === "sv" ? layout.physicalGroupChannels : layout.channelsPerGroup;
    var groupHeight = frame.h / Math.max(groupCount, 1);

    for (var group = 0; group < groupCount; group += 1) {
      var channelStart = kernel.kind === "qk" ? 0 : kernel.kind === "sv" ? group * layout.physicalGroupChannels : group * layout.channelsPerGroup;
      channelStart = channelStart % 32;
      var groupColor = channelColor(kernel, layout, channelStart, color);
      var gy = frame.y + group * groupHeight;
      append(svg, "rect", {
        x: frame.x,
        y: gy,
        width: frame.w,
        height: groupHeight,
        fill: groupColor.fill,
        opacity: group === 0 ? 0.68 : 0.42,
        "class": "matrix-channel-group-fill"
      });

      var visibleChannels = Math.max(1, Math.min(channelsInGroup, 32));
      for (var slice = 0; slice < visibleChannels; slice += 1) {
        var sx = frame.x + slice * frame.w / visibleChannels;
        append(svg, "rect", {
          x: sx,
          y: gy,
          width: frame.w / visibleChannels,
          height: groupHeight,
          fill: groupColor.strong,
          opacity: 0.045 + (slice % 3) * 0.035,
          "class": "matrix-channel-slice",
          "data-channel-slice": (channelStart + slice) % 32,
          "data-channel-group": group
        });
        if (slice > 0) {
          append(svg, "line", { x1: sx, y1: gy, x2: sx, y2: gy + groupHeight, "class": "matrix-grid-line fine" });
        }
        if (group === 0 && frame.w / visibleChannels >= 18 && groupHeight >= 12) {
          append(svg, "text", {
            x: sx + frame.w / visibleChannels / 2,
            y: gy + Math.min(groupHeight - 3, 10),
            fill: groupColor.text,
            "font-size": 5.8,
            "font-weight": 720,
            "text-anchor": "middle",
            "data-channel-label": (channelStart + slice) % 32
          }, "C" + ((channelStart + slice) % 32));
        }
      }

      if (group > 0) {
        append(svg, "line", {
          x1: frame.x,
          y1: gy,
          x2: frame.x + frame.w,
          y2: gy,
          "class": "matrix-grid-line"
        });
      }
      append(svg, "line", {
        x1: frame.x - 4,
        y1: gy + 1,
        x2: frame.x - 4,
        y2: gy + groupHeight - 1,
        stroke: groupColor.strong,
        "stroke-width": group === 0 ? 3.2 : 1.5,
        "class": "channel-group-rail",
        "data-channel-group": group
      });
      append(svg, "rect", {
        x: frame.x,
        y: gy,
        width: frame.w,
        height: groupHeight,
        fill: "transparent",
        "class": "channel-group-hit",
        "data-channel-group-outline": group
      });

      if (groupHeight >= 18 || group === 0 || group === groupCount - 1) {
        var lastChannel = (channelStart + channelsInGroup - 1) % 32;
        append(svg, "text", {
          x: frame.x + frame.w - 5,
          y: gy + Math.min(groupHeight - 3, 12),
          fill: groupColor.text,
          "font-size": groupHeight < 18 ? 6.5 : 8,
          "font-weight": 790,
          "text-anchor": "end",
          "class": "channel-group-label"
        }, "G" + group + " · C" + channelStart + (channelsInGroup > 1 ? "–C" + lastChannel : ""));
      }
    }

    append(svg, "line", { x1: frame.x, y1: frame.y - 15, x2: frame.x + 22, y2: frame.y - 15, stroke: color.strong, "stroke-width": 3 });
    append(svg, "text", { x: frame.x + 28, y: frame.y - 12, "class": "svg-small" }, "same color band + rail = one channel group");
  }

  function drawReductionSplitMatrixGrid(svg, frame, kernel, layout, color) {
    var bandHeight = frame.h / 32;
    for (var channel = 0; channel < 32; channel += 1) {
      var y = frame.y + channel * bandHeight;
      var channelTone = channelColor(kernel, layout, channel, color);
      append(svg, "rect", {
        x: frame.x,
        y: y,
        width: frame.w,
        height: bandHeight,
        fill: channelTone.fill,
        opacity: channel === 0 ? 0.8 : 0.34,
        "class": "matrix-channel-slice reduction-channel-band",
        "data-channel-slice": channel,
        "data-channel-group": channel
      });
      if (channel > 0) {
        append(svg, "line", { x1: frame.x, y1: y, x2: frame.x + frame.w, y2: y, "class": "matrix-grid-line fine" });
      }
      append(svg, "line", {
        x1: frame.x - 4,
        y1: y + 0.4,
        x2: frame.x - 4,
        y2: y + Math.max(0.6, bandHeight - 0.4),
        stroke: channelTone.strong,
        "stroke-width": channel === 0 ? 3.2 : 1.2,
        "class": "channel-group-rail",
        "data-channel-group": channel
      });
      append(svg, "rect", {
        x: frame.x,
        y: y,
        width: frame.w,
        height: bandHeight,
        fill: "transparent",
        "class": "channel-group-hit",
        "data-channel-group-outline": channel
      });
      if ((channel < 3 || channel === 31) && bandHeight >= 5.5) {
        append(svg, "text", {
          x: frame.x + frame.w - 4,
          y: y + bandHeight / 2 + 2,
          fill: channelTone.text,
          "font-size": Math.min(6, bandHeight * 0.72),
          "font-weight": 790,
          "text-anchor": "end",
          "class": "channel-group-label"
        }, "C" + channel + " · K/32 · full N");
      }
    }
    append(svg, "line", { x1: frame.x, y1: frame.y - 15, x2: frame.x + 22, y2: frame.y - 15, stroke: color.strong, "stroke-width": 3 });
    append(svg, "text", { x: frame.x + 28, y: frame.y - 12, "class": "svg-small" }, "32 channel bands split K · every channel owns full N");
  }

  function vectorGroupCount(kernel, layout) {
    if (state.mapping === "reduction_split") return 32;
    if (kernel.kind === "qk") return 1;
    if (kernel.kind === "sv") return Math.min(layout.contextGroups, 32);
    return Math.min(layout.reductionGroups, 32);
  }

  function drawGemvVector(svg, frame, kernel, layout, color, runtimeGeometry, canvas) {
    var inputX = canvas.vector.x;
    var inputW = canvas.vector.width;
    var y = canvas.vector.y;
    var attention = kernel.kind === "qk" || kernel.kind === "sv";
    var operandRows = attention ? 8 : BATCH_SIZE;
    var rowH = 6;
    var rowGap = 1.4;
    var waveGap = attention ? 5 : 0;
    var activeH = BATCH_SIZE * rowH + (BATCH_SIZE - 1) * rowGap;
    var h = operandRows * rowH + (operandRows - 1) * rowGap + waveGap;
    var groups = Math.max(1, vectorGroupCount(kernel, layout));
    var segmentW = inputW / groups;
    function operandRowY(row) {
      return y + row * (rowH + rowGap) + (attention && row >= BATCH_SIZE ? waveGap : 0);
    }
    append(svg, "text", { x: inputX - 9, y: y + h / 2 + 3, "class": "svg-mono", "text-anchor": "end" }, kernel.kind === "dense" ? "X" : "B0");
    if (attention) {
      append(svg, "text", { x: inputX, y: y - 19, "class": "svg-kicker" }, "FULL THIN GEMM OPERAND · B0 [8 × " + number(physicalMatrixShape(kernel).k) + "] · 8 GQA HEADS");
      append(svg, "rect", { x: inputX - 3, y: y - 3, width: inputW + 6, height: activeH + 6, rx: 4, "class": "gqa-wave-frame is-active" });
      append(svg, "rect", { x: inputX - 3, y: operandRowY(BATCH_SIZE) - 3, width: inputW + 6, height: activeH + 6, rx: 4, "class": "gqa-wave-frame is-next" });
    }
    for (var batchRow = 0; batchRow < operandRows; batchRow += 1) {
      var rowY = operandRowY(batchRow);
      for (var group = 0; group < groups; group += 1) {
        var channel = state.mapping === "reduction_split"
          ? group
          : kernel.kind === "sv"
            ? group * layout.physicalGroupChannels
            : kernel.kind === "qk"
              ? 0
              : group * layout.channelsPerGroup;
        channel %= 32;
        var segmentColor = channelColor(kernel, layout, channel, color);
        append(svg, "rect", {
          x: inputX + group * segmentW,
          y: rowY,
          width: segmentW,
          height: rowH,
          rx: groups === 1 ? 2 : 0.8,
          fill: segmentColor.fill,
          stroke: segmentColor.strong,
          "stroke-width": group === 0 ? 1.1 : 0.55,
          "class": "gemv-vector-row" + (attention && batchRow >= BATCH_SIZE ? " is-next-wave" : ""),
          "data-vector-slice": group,
          "data-channel-group": group,
          "data-batch-row": batchRow
        });
      }
      append(svg, "text", { x: inputX - 13, y: rowY + 5.5, fill: "#7b8799", "font-size": 5.8, "text-anchor": "end", "class": attention && batchRow >= BATCH_SIZE ? "next-wave-label" : "" }, kernel.kind === "dense" ? "B" + batchRow : "H" + batchRow);
    }
    if (attention) {
      append(svg, "text", { x: inputX + inputW + 9, y: y + activeH / 2 + 3, "class": "gqa-wave-label is-active" }, "GQA-A → GB");
      append(svg, "text", { x: inputX + inputW + 9, y: operandRowY(BATCH_SIZE) + activeH / 2 + 3, "class": "gqa-wave-label" }, "GQA-B next");
    } else {
      append(svg, "text", { x: inputX + inputW + 9, y: y + h / 2 + 3, "class": "svg-mono" }, "[4 × " + number(physicalMatrixShape(kernel).k) + "]");
    }
    append(svg, "path", {
      d: "M " + (inputX + inputW / 2) + " " + (y + h + 5) + " C " + (inputX + inputW / 2) + " " + (frame.y - 9) + ", " + (frame.x - 18) + " " + (frame.y - 9) + ", " + (frame.x - 7) + " " + (frame.y + frame.h / 2),
      stroke: "#91a0b5",
      "stroke-width": 1.2,
      fill: "none",
      "marker-end": "url(#arrowhead)"
    });
    append(svg, "text", { x: frame.x - 11, y: frame.y + frame.h / 2 - 8, "class": "svg-small", "text-anchor": "end" }, "stream along K rows");

    var groupK = layout.channelTileK;
    var chunkK = gbElementsPerRow(kernel, layout);
    var selectedW = Math.max(5, segmentW * Math.min(chunkK, groupK) / Math.max(groupK, 1));
    append(svg, "rect", {
      x: inputX,
      y: y - 2,
      width: selectedW,
      height: activeH + 4,
      rx: 3,
      "class": "selected-vector-slice",
      "data-selected-vector-k": chunkK
    });
    append(svg, "text", { x: inputX, y: y - 7, fill: "#b66a1d", "font-size": 7.5, "font-weight": 790 }, kernel.kind === "dense" ? "GB payload · 4 × " + number(chunkK) + " " + layout.precision + " values" : "GB TILE · GQA-A/H0–H3 · 4 × " + number(chunkK) + " " + layout.precision);
    runtimeGeometry.gb.slots.forEach(function (target, packetRow) {
      var sourceY = operandRowY(packetRow);
      var vectorPacket = append(svg, "rect", {
        x: inputX,
        y: sourceY,
        width: selectedW,
        height: rowH,
        rx: 1.5,
        "class": "vector-packet",
        "data-vector-packet": packetRow,
        "data-sx": inputX,
        "data-sy": sourceY,
        "data-sw": selectedW,
        "data-sh": rowH,
        "data-tx": target.x,
        "data-ty": target.y,
        "data-tw": target.w,
        "data-th": target.h
      });
      vectorPacket.style.fill = color.strong;
    });
    append(svg, "path", {
      d: "M " + (inputX + selectedW / 2) + " " + (y - 4) + " C 530 125, 570 575, " + (runtimeGeometry.gb.x - 8) + " " + (runtimeGeometry.gb.y + runtimeGeometry.gb.h / 2),
      "class": "vector-to-gb-path",
      id: "vectorToGbPath"
    });
  }

  function selectedTile(frame, kernel, layout) {
    if (state.mapping === "reduction_split") {
      return {
        x: frame.x,
        y: frame.y,
        w: frame.w,
        h: frame.h / 32,
        channelCount: 1,
        labelOutside: frame.h / 32 < 11,
        groupSummary: "Selected reduction set · " + number(layout.groupTileK) + " × " + number(layout.groupTileN),
        channelSummary: "Selected C0 · " + number(layout.channelTileK) + " × " + number(layout.channelTileN)
      };
    }
    if (kernel.kind === "qk") {
      return {
        x: frame.x,
        y: frame.y,
        w: frame.w / layout.channelsPerHead,
        h: frame.h,
        channelCount: 1,
        groupSummary: "Selected G0 / one KV head · " + number(layout.groupTileK) + " × " + contextLabel(state.context),
        channelSummary: "Selected C0 · " + number(layout.channelTileK) + " × " + number(layout.channelTileN)
      };
    }
    if (kernel.kind === "sv") {
      return {
        x: frame.x,
        y: frame.y,
        w: frame.w,
        h: frame.h / Math.max(layout.contextGroups, 1),
        channelCount: layout.physicalGroupChannels,
        groupSummary: "Selected G0 · " + number(layout.groupTileK) + " × " + number(layout.groupTileN),
        channelSummary: "Selected C0 · " + number(layout.channelTileK) + " × " + number(layout.channelTileN)
      };
    }
    var tileFraction = Math.min(1, layout.outputCapacity / Math.max(layout.localN, 1));
    var groupFraction = 1 / Math.max(layout.reductionGroups, 1);
    return {
      x: frame.x,
      y: frame.y,
      w: frame.w * tileFraction,
      h: frame.h * groupFraction,
      channelCount: layout.channelsPerGroup,
      groupSummary: "Selected G0 · " + number(layout.groupTileK) + " × " + number(layout.groupTileN),
      channelSummary: "Selected C0 · " + number(layout.channelTileK) + " × " + number(layout.channelTileN)
    };
  }

  function drawSelectedTile(svg, selected, kernel, layout, color, canvas) {
    var tileGroup = append(svg, "g", {
      "class": "matrix-tile-clickable",
      role: "button",
      tabindex: "0",
      "aria-label": "Map selected matrix tile to DRAM banks"
    });
    var channelCount = Math.max(1, Math.min(selected.channelCount || 1, 32));
    var channelWidth = selected.w / channelCount;
    for (var channel = 0; channel < channelCount; channel += 1) {
      var channelX = selected.x + channel * channelWidth;
      var channelTone = channelColor(kernel, layout, channel, color);
      append(tileGroup, "rect", {
        x: channelX,
        y: selected.y,
        width: channelWidth,
        height: selected.h,
        fill: channelTone.fill,
        "class": "selected-tile-channel" + (channel === 0 ? " is-primary" : ""),
        "data-channel-group": 0,
        "data-selected-channel": channel
      });
      if (channel > 0) {
        append(tileGroup, "line", {
          x1: channelX,
          y1: selected.y,
          x2: channelX,
          y2: selected.y + selected.h,
          "class": "selected-tile-divider"
        });
      }
      if (!selected.labelOutside) {
        append(tileGroup, "text", {
          x: channelX + channelWidth / 2,
          y: selected.y + selected.h / 2 + 3,
          fill: channelTone.text,
          "font-size": channelCount > 16 ? 5.2 : channelCount > 8 ? 5.8 : 7,
          "font-weight": 820,
          "text-anchor": "middle",
          "class": "selected-channel-label"
        }, "C" + channel);
      }
    }
    append(tileGroup, "rect", {
      x: selected.x,
      y: selected.y,
      width: selected.w,
      height: selected.h,
      "class": "selected-tile"
    });
    if (selected.labelOutside) {
      append(tileGroup, "text", {
        x: selected.x + selected.w + 7,
        y: selected.y + Math.max(7, selected.h / 2 + 3),
        "class": "selected-channel-callout"
      }, "C0 · full N");
    }
    append(tileGroup, "text", {
      x: canvas.vector.x,
      y: canvas.tileMetaY,
      "class": "selected-tile-summary"
    }, selected.groupSummary);
    append(tileGroup, "text", {
      x: canvas.vector.x,
      y: canvas.tileMetaY + 16,
      "class": "selected-channel-summary"
    }, selected.channelSummary);
    append(tileGroup, "rect", {
      x: canvas.tileAction.x,
      y: canvas.tileAction.y,
      width: canvas.tileAction.width,
      height: canvas.tileAction.height,
      rx: 8,
      "class": "tile-play-pill"
    });
    append(tileGroup, "text", {
      x: canvas.tileAction.x + canvas.tileAction.width / 2,
      y: canvas.tileAction.y + 17,
      "class": "tile-play-label",
      "text-anchor": "middle"
    }, "CLICK · PLAY 1→4");
    tileGroup.addEventListener("click", revealPhysicalMapping);
    tileGroup.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        revealPhysicalMapping();
      }
    });
  }

  function waveShapeLabel(layout, remainingK, remainingN) {
    var activeN = Math.min(layout.channelWaveWidth, remainingN);
    if (layout.svPartitionMode === "k_chunks") {
      var partSizes = [];
      for (var part = 0; part < layout.parallelKParts; part += 1) {
        var partK = Math.min(layout.kChunkCapacity, Math.max(0, remainingK - part * layout.kChunkCapacity));
        if (partK > 0) partSizes.push(partK);
      }
      if (partSizes.length > 1 && partSizes.every(function (size) { return size === partSizes[0]; })) {
        return partSizes.length + "×(" + number(partSizes[0]) + "×" + number(activeN) + ")";
      }
      return "K(" + partSizes.map(number).join("+") + ")×" + number(activeN);
    }
    if (layout.svPartitionMode === "output_slices") {
      return number(Math.min(layout.kChunkCapacity, remainingK)) + "×" + number(activeN) + " · N/2×2";
    }
    return number(Math.min(layout.kChunkCapacity, remainingK)) + "×" + number(activeN);
  }

  function drawWavePanel(svg, kernel, layout, color, canvas) {
    var panel = canvas.wavePanel;
    var group = append(svg, "g", {
      id: "wavePanel",
      "class": "qk-wave-panel",
      "data-wave-count": layout.totalWaves,
      "data-wave-k": layout.waveK,
      "data-wave-n": layout.waveN,
      "data-channel-k": layout.channelTileK,
      "data-channel-n": layout.channelTileN,
      "data-output-waves": layout.outputWaves,
      "data-k-chunks": layout.kChunks,
      "data-k-capacity": layout.kChunkCapacity,
      "data-k-coverage": layout.kCoveragePerWave,
      "data-parallel-k-parts": layout.parallelKParts,
      "data-sv-partition-mode": layout.svPartitionMode || "none",
      "data-n-capacity": layout.channelWaveWidth,
      "data-reduction-split": state.mapping === "reduction_split" ? "true" : "false"
    });
    append(group, "rect", { x: panel.x, y: panel.y, width: panel.width, height: panel.height, rx: 12, "class": "qk-wave-panel-bg" });
    append(group, "text", { x: panel.x + 16, y: panel.y + 18, "class": "svg-kicker" }, "C0 WAVE DECOMPOSITION");
    append(group, "text", { x: panel.x + panel.width - 16, y: panel.y + 18, "class": "qk-wave-bank-note", "text-anchor": "end" },
      kernel.kind === "sv"
        ? layout.physicalSaPartitions + " × SA 4×" + layout.saPartitionWidth + " · " + layout.banksPerChannel + " banks in lockstep"
        : layout.banksPerChannel + " banks/wave · bank tile " + number(layout.waveK) + "×" + layout.burstLength);
    append(group, "text", { x: panel.x + 16, y: panel.y + 34, "class": "qk-wave-subtitle" },
      "C0 [" + number(layout.channelTileK) + " × " + number(layout.channelTileN) + "] → " + layout.outputWaves + " N wave(s) × " + layout.kChunks + " physical K wave(s) = " + number(layout.totalWaves));

    var cardX = panel.x + 16;
    var cardY = panel.y + 43;
    var cardGap = 6;
    var cardWidth = (panel.width - 32 - cardGap * 3) / 4;
    var cardHeight = 45;
    for (var slot = 0; slot < 4; slot += 1) {
      var x = cardX + slot * (cardWidth + cardGap);
      var card = append(group, "g", { "class": "qk-wave-slot", "data-wave-slot": slot });
      append(card, "rect", { x: x, y: cardY, width: cardWidth, height: cardHeight, rx: 5, "class": "qk-wave-card" });
      append(card, "text", { x: x + 8, y: cardY + 14, "class": "qk-wave-title", "data-wave-title": slot }, "W" + slot);
      append(card, "text", { x: x + 8, y: cardY + 28, "class": "qk-wave-shape", "data-wave-shape": slot }, waveShapeLabel(layout, layout.channelTileK, layout.channelTileN));
      var laneY = cardY + 34;
      var laneW = (cardWidth - 12) / layout.banksPerChannel;
      for (var bank = 0; bank < layout.banksPerChannel; bank += 1) {
        append(card, "rect", {
          x: x + 6 + bank * laneW,
          y: laneY,
          width: laneW,
          height: 6,
          fill: bank === 0 ? color.strong : color.fill,
          "class": "qk-wave-bank-lane",
          "data-wave-bank": bank
        });
      }
    }

    var ticksX = panel.x + 16;
    var ticksY = panel.y + 95;
    var ticksWidth = panel.width - 32;
    var displayTicks = Math.min(layout.totalWaves, 128);
    var tickGap = displayTicks > 64 ? 0.5 : displayTicks > 24 ? 1 : 2;
    var tickWidth = Math.max(0.7, (ticksWidth - tickGap * (displayTicks - 1)) / displayTicks);
    group.setAttribute("data-progress-ticks", displayTicks);
    for (var tickIndex = 0; tickIndex < displayTicks; tickIndex += 1) {
      var waveStart = Math.floor(tickIndex * layout.totalWaves / displayTicks);
      var waveEnd = Math.floor((tickIndex + 1) * layout.totalWaves / displayTicks);
      append(group, "rect", {
        x: ticksX + tickIndex * (tickWidth + tickGap),
        y: ticksY,
        width: tickWidth,
        height: 5,
        rx: 1,
        "class": "qk-wave-tick",
        "data-wave-tick": tickIndex,
        "data-wave-start": waveStart,
        "data-wave-end": waveEnd
      });
    }
    append(group, "text", { x: panel.x + 16, y: panel.y + 113, "class": "qk-wave-status", id: "waveStatus" }, "Ready · 0/" + number(layout.totalWaves) + " waves accumulated");
    append(group, "text", { x: panel.x + panel.width - 16, y: panel.y + 113, "class": "qk-wave-page", id: "wavePage", "text-anchor": "end" }, "page 1/" + Math.ceil(layout.totalWaves / 4));

    var phase = append(group, "g", {
      "class": "phase-button qk-wave-phase",
      "data-phase-button": 4,
      "data-phase": 4,
      role: "button",
      tabindex: "0",
      "aria-label": "Play phase 4: stream C0 waves through " + saLabel()
    });
    append(phase, "rect", { x: panel.x + 16, y: panel.y + 121, width: panel.width - 32, height: 28, rx: 8, "class": "phase-pill" });
    append(phase, "text", { x: panel.x + 30, y: panel.y + 139, "class": "phase-text" }, state.mapping === "reduction_split" ? "4 · stream C0 waves · 32 channels in lockstep · Σ C0…C31" : "4 · stream C0 waves through " + saLabel() + " · " + layout.banksPerChannel + " banks in parallel");
    phase.addEventListener("click", function () { playPhysicalStage(4); });
    phase.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        playPhysicalStage(4);
      }
    });
    renderWaveState(-1, 0);
  }

  function renderWaveState(activeWave, completedWaves) {
    var panel = document.getElementById("wavePanel");
    if (!panel) return;
    var waveCount = Number(panel.getAttribute("data-wave-count"));
    var channelK = Number(panel.getAttribute("data-channel-k"));
    var channelN = Number(panel.getAttribute("data-channel-n"));
    var outputWaves = Number(panel.getAttribute("data-output-waves"));
    var kChunks = Number(panel.getAttribute("data-k-chunks"));
    var kCapacity = Number(panel.getAttribute("data-k-capacity"));
    var kCoverage = Number(panel.getAttribute("data-k-coverage"));
    var parallelKParts = Number(panel.getAttribute("data-parallel-k-parts"));
    var svPartitionMode = panel.getAttribute("data-sv-partition-mode");
    var nCapacity = Number(panel.getAttribute("data-n-capacity"));
    var focusWave = activeWave >= 0 ? activeWave : Math.min(completedWaves, waveCount - 1);
    var pageStart = Math.max(0, Math.floor(focusWave / 4) * 4);
    panel.querySelectorAll("[data-wave-slot]").forEach(function (slot) {
      var slotIndex = Number(slot.getAttribute("data-wave-slot"));
      var waveIndex = pageStart + slotIndex;
      var visible = waveIndex < waveCount;
      slot.classList.toggle("is-hidden", !visible);
      slot.classList.toggle("is-active", visible && waveIndex === activeWave);
      slot.classList.toggle("is-complete", visible && waveIndex < completedWaves);
      slot.setAttribute("data-wave-index", waveIndex);
      var title = slot.querySelector("[data-wave-title]");
      var shape = slot.querySelector("[data-wave-shape]");
      if (title) title.textContent = "W" + waveIndex;
      if (shape && visible) {
        var nIndex = waveIndex % outputWaves;
        var kIndex = Math.floor(waveIndex / outputWaves);
        var remainingK = Math.max(0, channelK - kIndex * kCoverage);
        var remainingN = Math.max(0, channelN - nIndex * nCapacity);
        shape.textContent = waveShapeLabel({
          channelWaveWidth: nCapacity,
          kChunkCapacity: kCapacity,
          parallelKParts: parallelKParts,
          svPartitionMode: svPartitionMode
        }, remainingK, remainingN);
      } else if (shape) {
        shape.textContent = "";
      }
    });
    panel.querySelectorAll("[data-wave-tick]").forEach(function (tick) {
      var waveStart = Number(tick.getAttribute("data-wave-start"));
      var waveEnd = Number(tick.getAttribute("data-wave-end"));
      tick.classList.toggle("is-active", activeWave >= waveStart && activeWave < waveEnd);
      tick.classList.toggle("is-complete", waveEnd <= completedWaves);
    });
    var page = document.getElementById("wavePage");
    if (page) page.textContent = "page " + (Math.floor(pageStart / 4) + 1) + "/" + Math.ceil(waveCount / 4);
    var status = document.getElementById("waveStatus");
    if (!status) return;
    if (completedWaves >= waveCount) {
      status.textContent = panel.getAttribute("data-reduction-split") === "true" ? "C0 complete · C0…C31 partials reduce into the device output" : "C0 complete · companion channels/groups follow the same schedule";
    } else if (activeWave >= 0) {
      var activeN = activeWave % outputWaves;
      var activeK = Math.floor(activeWave / outputWaves);
      status.textContent = "Wave " + (activeWave + 1) + "/" + waveCount + " · K" + (activeK + 1) + "/" + kChunks + " · N" + (activeN + 1) + "/" + outputWaves;
    } else {
      status.textContent = "Ready · " + completedWaves + "/" + waveCount + " waves accumulated";
    }
  }

  function bankPanelSubtitle(kernel, layout) {
    if (state.mapping === "reduction_split") {
      return "full N/channel · 32-way K split · local partial reduction";
    }
    if (kernel.kind === "qk") {
      return layout.banksPerChannel + " banks/selected channel · C0 owns " + number(layout.contextsPerChannel) + " context columns";
    }
    if (kernel.kind === "sv") {
      return layout.physicalGroupChannels + " channel(s)/128-dim group · " + svPartitionSummary(layout);
    }
    return layout.channelsPerGroup + " channel(s)/group · " + layout.reductionGroups + " reduction group(s)";
  }

  function activeBank(kernel, layout, channel) {
    if (state.mapping === "reduction_split") return true;
    if (kernel.kind === "qk") return channel === 0;
    if (kernel.kind === "sv") return true;
    return channel < layout.activeChannels;
  }

  function channelColor(kernel, layout, channel, deviceColor) {
    if (state.mapping === "reduction_split") {
      return DEVICE_COLORS[(state.device + channel) % DEVICE_COLORS.length];
    }
    if (kernel.kind === "qk") {
      var head = Math.floor(channel / layout.channelsPerHead);
      return DEVICE_COLORS[(state.device + head) % DEVICE_COLORS.length];
    }
    if (kernel.kind === "sv") {
      var group = Math.floor(channel / layout.physicalGroupChannels);
      return DEVICE_COLORS[(state.device + group) % DEVICE_COLORS.length];
    }
    if (layout.reductionGroups > 1) {
      var reductionGroup = Math.floor(channel / layout.channelsPerGroup);
      return DEVICE_COLORS[(state.device + reductionGroup) % DEVICE_COLORS.length];
    }
    return deviceColor;
  }

  function drawBankGrid(parent, kernel, layout, deviceColor) {
    var x = 685;
    var y = 140;
    var width = 580;
    var rowHeight = 10.8;
    var gap = 2;
    var cellW = (width - gap * (layout.banksPerChannel - 1)) / layout.banksPerChannel;
    var targets = [];

    append(parent, "text", { x: 615, y: y - 13, "class": "svg-small" }, state.mapping === "reduction_split" ? "CHANNEL / K SLICE" : "CHANNEL / GROUP");
    for (var headerBank = 0; headerBank < layout.banksPerChannel; headerBank += 1) {
      if (headerBank < 2 || headerBank === layout.banksPerChannel - 1) {
        append(parent, "text", {
          x: x + headerBank * (cellW + gap) + cellW / 2,
          y: y - 13,
          "class": "svg-small",
          "text-anchor": "middle"
        }, headerBank === layout.banksPerChannel - 1 && headerBank > 2 ? "B" + headerBank : "B" + headerBank);
      }
      if (headerBank === 2 && layout.banksPerChannel > 4) {
        append(parent, "text", { x: x + 2.5 * (cellW + gap), y: y - 13, "class": "svg-small", "text-anchor": "middle" }, "…");
      }
    }

    for (var channel = 0; channel < 32; channel += 1) {
      var cy = y + channel * rowHeight;
      var groupColor = channelColor(kernel, layout, channel, deviceColor);
      append(parent, "text", { x: 650, y: cy + 7.5, "class": "svg-small", "text-anchor": "end" }, "C" + channel);
      append(parent, "rect", { x: 657, y: cy, width: 8, height: 8.6, rx: 2, fill: groupColor.strong, opacity: activeBank(kernel, layout, channel) ? 0.82 : 0.16 });
      append(parent, "rect", { x: x - 4, y: cy - 1, width: width + 8, height: 10.5, rx: 2, "class": "channel-strip" });
      for (var bank = 0; bank < layout.banksPerChannel; bank += 1) {
        var bx = x + bank * (cellW + gap);
        var cell = append(parent, "rect", {
          x: bx,
          y: cy,
          width: cellW,
          height: 8.6,
          rx: 1.5,
          "class": "bank-cell",
          "data-active-bank": activeBank(kernel, layout, channel) ? "true" : "false",
          style: "--bank-color:" + groupColor.fill + ";--bank-stroke:" + groupColor.strong
        });
        if (layout.representativeChannels.indexOf(channel) >= 0) {
          targets.push({ x: bx, y: cy, w: cellW, h: 8.6, node: cell, channel: channel, bank: bank });
        }
      }
    }

    var channelGroups;
    var groupSize;
    if (state.mapping === "reduction_split") {
      channelGroups = 32;
      groupSize = 1;
    } else if (kernel.kind === "qk") {
      channelGroups = layout.localKvHeads;
      groupSize = layout.channelsPerHead;
    } else if (kernel.kind === "sv") {
      channelGroups = Math.floor(32 / layout.physicalGroupChannels);
      groupSize = layout.physicalGroupChannels;
    } else {
      channelGroups = layout.reductionGroups;
      groupSize = layout.channelsPerGroup;
    }
    channelGroups = Math.max(1, Math.min(channelGroups, Math.ceil(32 / Math.max(groupSize, 1))));
    for (var group = 0; group < channelGroups; group += 1) {
      var firstChannel = group * groupSize;
      if (firstChannel >= 32) break;
      var groupColor = channelColor(kernel, layout, firstChannel, deviceColor);
      var groupHeight = Math.min(groupSize, 32 - firstChannel) * rowHeight;
      append(parent, "rect", {
        x: 672,
        y: y + firstChannel * rowHeight - 2,
        width: width + 17,
        height: groupHeight + 1,
        rx: 4,
        fill: "none",
        stroke: groupColor.strong,
        "stroke-width": group === 0 ? 2 : 0.9,
        opacity: group === 0 ? 0.9 : 0.48,
        "class": "physical-channel-group" + (group === 0 ? " is-highlighted" : ""),
        "data-physical-group": group
      });
      if (groupHeight >= 20 || group === 0) {
        append(parent, "text", {
          x: 675,
          y: y + firstChannel * rowHeight + Math.min(12, groupHeight / 2 + 3),
          fill: groupColor.text,
          "font-size": 7,
          "font-weight": 800,
          "text-anchor": "end"
        }, "G" + group);
      }
    }
    return { targets: targets, x: x, y: y, width: width, cellW: cellW, rowHeight: rowHeight };
  }

  function svSaPartitionLabels(layout) {
    if (layout.svPartitionMode === "k_chunks") return ["K-part A", "K-part B"];
    if (layout.svPartitionMode === "output_slices") return ["N-left", "N-right"];
    return ["active", "idle"];
  }

  function drawRuntimePanel(parent, kernel, layout) {
    var runtime = append(parent, "g", { id: "runtimePanel", "class": "runtime-panel" });
    var gb = { x: 638, y: 548, w: 218, h: 78, slots: [] };
    var sa = { x: 910, y: 548, w: 205, h: 78 };
    append(runtime, "line", { x1: 615, y1: 508, x2: 1285, y2: 508, stroke: "#d5dde8" });
    append(runtime, "text", { x: 615, y: 531, "class": "svg-kicker" }, "RUNTIME DATAFLOW · BANK WORD = " + layout.burstLength + " " + layout.precision + " VALUES");

    append(runtime, "rect", { x: gb.x, y: gb.y, width: gb.w, height: gb.h, rx: 10, "class": "gb-box" });
    append(runtime, "text", { x: gb.x + gb.w / 2, y: gb.y + 16, fill: "#7d4b14", "font-size": 9.5, "font-weight": 780, "text-anchor": "middle" }, "GLOBAL BUFFER · 2 KiB");
    var slotX = gb.x + 31;
    var slotY = gb.y + 23;
    var slotW = gb.w - 43;
    var slotH = 7;
    for (var gbRow = 0; gbRow < BATCH_SIZE; gbRow += 1) {
      var rowY = slotY + gbRow * 9.5;
      append(runtime, "text", { x: gb.x + 25, y: rowY + 6, fill: "#9a6427", "font-size": 6.8, "font-weight": 760, "text-anchor": "end" }, kernel.kind === "dense" ? "B" + gbRow : "H" + gbRow);
      append(runtime, "rect", { x: slotX, y: rowY, width: slotW, height: slotH, rx: 2, "class": "gb-vector-slot", "data-gb-slot": gbRow });
      if (kernel.kind === "sv" && layout.gbInputStreams > 1) {
        for (var gbDivider = 1; gbDivider < layout.gbInputStreams; gbDivider += 1) {
          append(runtime, "line", { x1: slotX + slotW * gbDivider / layout.gbInputStreams, y1: rowY, x2: slotX + slotW * gbDivider / layout.gbInputStreams, y2: rowY + slotH, "class": "gb-half-divider" });
        }
      }
      gb.slots.push({ x: slotX, y: rowY, w: slotW, h: slotH });
    }
    append(runtime, "text", { x: gb.x + gb.w / 2, y: gb.y + 72, fill: "#9a6427", "font-size": kernel.kind === "sv" ? 6.3 : 7.2, "text-anchor": "middle" }, gbCapacityLabel(kernel, layout));

    append(runtime, "path", { d: "M " + (gb.x + gb.w) + " " + (gb.y + gb.h / 2) + " H " + (sa.x - 10), stroke: "#8d9caf", "stroke-width": 1.5, "marker-end": "url(#arrowhead)", "class": "runtime-arrow" });
    append(runtime, "path", { d: "M 1115 490 C 1115 525, 1035 523, 1035 " + (sa.y - 8), stroke: "#8d9caf", "stroke-width": 1.5, fill: "none", "marker-end": "url(#arrowhead)", "class": "runtime-arrow" });
    append(runtime, "text", { x: 1120, y: 523, fill: "#748196", "font-size": 7.5 }, kernel.kind === "sv"
      ? layout.saPartitionWidth + " V elements/half · broadcast across GQA H0–H3"
      : "bank-row stream");

    append(runtime, "rect", { x: sa.x, y: sa.y, width: sa.w, height: sa.h, rx: 10, "class": "sa-box" });
    append(runtime, "text", { x: sa.x + 11, y: sa.y + 17, fill: "#08675f", "font-size": 10, "font-weight": 790 }, saLabel());
    append(runtime, "text", { x: sa.x + sa.w - 10, y: sa.y + 17, fill: "#328f86", "font-size": 7.2, "text-anchor": "end" }, kernel.kind === "dense" ? "rows = B0…B3" : "B0 · GQA H0…H3");
    var cellStartX = sa.x + 11;
    var cellStartY = sa.y + 27;
    var cellGap = 1.2;
    var cellW = (sa.w - 22 - cellGap * (layout.systolicWidth - 1)) / layout.systolicWidth;
    var cellH = 7;
    for (var row = 0; row < 4; row += 1) {
      for (var column = 0; column < layout.systolicWidth; column += 1) {
        var saPartition = kernel.kind === "sv" ? Math.floor(column / layout.saPartitionWidth) : 0;
        var localSaColumn = kernel.kind === "sv" ? column % layout.saPartitionWidth : column;
        append(runtime, "rect", {
          x: cellStartX + column * (cellW + cellGap),
          y: cellStartY + row * (cellH + cellGap),
          width: cellW,
          height: cellH,
          rx: 1,
          "class": "sa-cell" + (kernel.kind === "sv" ? " sa-part-" + saPartition : ""),
          "data-sa-cell": row + "-" + column,
          "data-sa-partition": saPartition,
          style: "--stream-delay:" + (row * 34 + localSaColumn * 23) + "ms"
        });
      }
    }
    if (kernel.kind === "sv") {
      var partitionLabels = svSaPartitionLabels(layout);
      var gridHeight = 4 * cellH + 3 * cellGap;
      for (var saPart = 0; saPart < layout.physicalSaPartitions; saPart += 1) {
        var firstColumn = saPart * layout.saPartitionWidth;
        var columnsInPart = Math.min(layout.saPartitionWidth, layout.systolicWidth - firstColumn);
        var partitionX = cellStartX + firstColumn * (cellW + cellGap);
        var partitionW = columnsInPart * cellW + Math.max(0, columnsInPart - 1) * cellGap;
        append(runtime, "rect", {
          x: partitionX - 2,
          y: cellStartY - 2,
          width: partitionW + 4,
          height: gridHeight + 4,
          rx: 3,
          "class": "sa-partition-outline sa-partition-" + saPart
        });
        append(runtime, "text", {
          x: partitionX + partitionW / 2,
          y: sa.y + sa.h - 6,
          "text-anchor": "middle",
          "class": "sa-partition-label sa-partition-label-" + saPart
        }, partitionLabels[saPart] + " · 4×" + columnsInPart);
      }
    }

    append(runtime, "path", { d: "M " + (sa.x + sa.w) + " " + (sa.y + sa.h / 2) + " H 1154", stroke: "#8d9caf", "stroke-width": 1.5, "marker-end": "url(#arrowhead)" });
    append(runtime, "rect", { x: 1160, y: 566, width: 115, height: 36, rx: 18, fill: kernel.split === "row" ? "#efeafe" : "#e6f5f1", stroke: kernel.split === "row" ? "#8e77d5" : "#2b9b68" });
    append(runtime, "text", { x: 1217, y: 588, fill: kernel.split === "row" ? "#5a43ad" : "#14704c", "font-size": 8.5, "font-weight": 760, "text-anchor": "middle" }, resultReduction(kernel));
    return { gb: gb, sa: sa };
  }

  function gbCapacityLabel(kernel, layout) {
    if (kernel.kind === "sv") {
      if (layout.svPartitionMode === "k_chunks") return "B0/GQA-A · 4×(K-A " + layout.waveK + " + K-B " + layout.waveK + ") = 4×" + layout.kCoveragePerWave;
      if (layout.svPartitionMode === "output_slices") return "B0/GQA-A · 4×" + layout.waveK + " K values · broadcast to N halves";
      return "B0/GQA-A · 4×" + layout.waveK + " K values";
    }
    if (kernel.kind === "qk") return "B0/GQA-A · 4 heads × " + gbElementsPerRow(kernel, layout) + " values";
    return "4 batch rows × " + gbElementsPerRow(kernel, layout) + " " + layout.precision + " values";
  }

  function gbElementsPerRow(kernel, layout) {
    return layout.waveK;
  }

  function runtimeOperand(kernel) {
    if (kernel.kind === "qk") return "Batch 0 · four GQA heads";
    if (kernel.kind === "sv") return "Batch 0 · four GQA score rows";
    return "four activation-row K slices";
  }

  function resultReduction(kernel) {
    if (state.mapping === "reduction_split") {
      return kernel.split === "row" ? "LOCAL Σ → TP" : "LOCAL Σ C0…C31";
    }
    if (kernel.split === "row") return "TP PARTIAL";
    if (kernel.kind === "sv" || kernel.short === "Wq" || kernel.short === "[Wk | Wv]") return "LOCAL PNM";
    return "LOCAL OUTPUT";
  }

  function drawPackets(svg, selected, bankGeometry, layout, kernel, deviceColor) {
    var targets = bankGeometry.targets;
    if (!targets.length) return;
    targets.forEach(function (target, index) {
      var sourceW = Math.max(0.35, selected.w / targets.length);
      var sx = selected.x + index * selected.w / targets.length;
      var sy = selected.y + selected.h * 0.18;
      var node = append(svg, "rect", {
        x: sx,
        y: sy,
        width: sourceW,
        height: selected.h * 0.64,
        rx: 1.5,
        "class": "packet-tile",
        "data-packet": "true",
        "data-sx": sx,
        "data-sy": sy,
        "data-sw": sourceW,
        "data-sh": selected.h * 0.64,
        "data-tx": target.x,
        "data-ty": target.y,
        "data-tw": target.w,
        "data-th": target.h
      });
      node.style.fill = channelColor(kernel, layout, target.channel, deviceColor).fill;
    });
  }

  function drawExecutionPhases(svg, phaseY, kernel) {
    var labels = ["1 · select matrix tile", "2 · place tile in banks", kernel.kind === "qk" || kernel.kind === "sv" ? "3 · load GQA-A: H0–H3 → GB" : "3 · load 4 rows → GB", "4 · stream " + saLabel() + " + reduce"];
    var notes = ["reset", "replay", "replay", "replay"];
    var startX = 40;
    var visibleLabels = labels.slice(0, 3);
    var width = 418;
    visibleLabels.forEach(function (label, index) {
      var phase = index + 1;
      var x = startX + index * (width + 12);
      var group = append(svg, "g", {
        "class": "phase-button",
        "data-phase-button": phase,
        "data-phase": phase,
        role: "button",
        tabindex: "0",
        "aria-label": "Play phase " + phase + ": " + label
      });
      append(group, "rect", { x: x, y: phaseY, width: width, height: 48, rx: 10, "class": "phase-pill" });
      append(group, "text", { x: x + 15, y: phaseY + 21, "class": "phase-text" }, label);
      append(group, "text", { x: x + 15, y: phaseY + 37, "class": "phase-note" }, notes[index] + " this stage");
      (function (selectedPhase) {
        group.addEventListener("click", function () { playPhysicalStage(selectedPhase); });
        group.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            playPhysicalStage(selectedPhase);
          }
        });
      })(phase);
    });
  }

  function wireChannelGroupHighlight(svg) {
    var triggers = svg.querySelectorAll("[data-channel-group-outline], [data-physical-group]");
    function toggle(group, active) {
      svg.querySelectorAll("[data-channel-group], [data-channel-group-outline], [data-physical-group]").forEach(function (node) {
        var raw = node.getAttribute("data-channel-group");
        if (raw === null) raw = node.getAttribute("data-channel-group-outline");
        if (raw === null) raw = node.getAttribute("data-physical-group");
        if (Number(raw) === group) node.classList.toggle("is-linked", active);
      });
    }
    triggers.forEach(function (node) {
      var raw = node.getAttribute("data-channel-group-outline");
      if (raw === null) raw = node.getAttribute("data-physical-group");
      var group = Number(raw);
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "button");
      node.setAttribute("aria-label", "Highlight channel group " + group);
      node.addEventListener("mouseenter", function () { toggle(group, true); });
      node.addEventListener("mouseleave", function () { toggle(group, false); });
      node.addEventListener("focus", function () { toggle(group, true); });
      node.addEventListener("blur", function () { toggle(group, false); });
    });
  }

  function selectedRuleLines(kernel, layout) {
    if (state.mapping === "reduction_split") {
      return [
        "Each channel owns all " + number(layout.channelTileN) + " N columns and " + number(layout.channelTileK) + " of the K-reduction rows.",
        "C0 runs " + layout.outputWaves + " N wave(s) × " + layout.kChunks + " physical K wave(s) = " + number(layout.totalWaves) + "; " +
          (kernel.kind === "sv" ? svPartitionSummary(layout) + "; " : "") +
          "then Σ C0…C31" + (kernel.split === "row" ? " feeds the TP All-Reduce." : " finishes the device-local output.")
      ];
    }
    if (kernel.kind === "qk") {
      return [
        "One KV head is sequence-striped across " + layout.channelsPerHead + " channels; selected C0 owns 128 × " + number(layout.contextsPerChannel) + ".",
        "Each C0 wave covers 128 × " + number(layout.channelWaveWidth) + " across " + layout.banksPerChannel + " banks; " + layout.wavesPerChannel + " wave(s) complete C0."
      ];
    }
    if (kernel.kind === "sv") {
      return [
        "One KV head uses " + layout.channelsPerHead + " channels; C0 owns " + number(layout.channelTileN) + " N outputs and streams four GQA heads together.",
        svPartitionSummary(layout) + "; each fetched " + layout.saPartitionWidth + "-element V slice broadcasts across H0–H3."
      ];
    }
    return [
      "K occupies rows; N is divided into " + layout.burstLength + "-column " + layout.precision + " bank words.",
      layout.reductionGroups > 1 ? "The same output tile is accumulated across " + layout.reductionGroups + " channel reduction groups." : "One device tile uses up to " + number(layout.totalBanks) + " banks in parallel."
    ];
  }

  function revealPhysicalMapping() {
    var token = ++state.animationToken;
    resetPhysicalVisuals();
    revealBankPanel();
    document.getElementById("physicalHint").textContent = "Phase 2/4 · tile slices are moving to their channel banks";
    animateBankPlacement(token, function () {
      setPhase(2);
      schedule(token, reducedMotion() ? 1 : 260, function () {
        document.getElementById("runtimePanel").classList.add("is-active");
        document.getElementById("physicalHint").textContent = "Phase 3/4 · selected vector K-slice is loading into GB";
        animateVectorToGb(token, function () {
          setPhase(3);
          schedule(token, reducedMotion() ? 1 : 260, function () {
            runPhaseFour(token);
          });
        });
      });
    });
  }

  function reducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function setAnimatedRect(node, t) {
    var eased = 1 - Math.pow(1 - t, 3);
    ["x", "y", "w", "h"].forEach(function (key) {
      var sourceKey = key === "w" ? "sw" : key === "h" ? "sh" : "s" + key;
      var targetKey = key === "w" ? "tw" : key === "h" ? "th" : "t" + key;
      var source = Number(node.getAttribute("data-" + sourceKey));
      var target = Number(node.getAttribute("data-" + targetKey));
      var attr = key === "w" ? "width" : key === "h" ? "height" : key;
      node.setAttribute(attr, source + (target - source) * eased);
    });
    node.style.opacity = t > 0 ? "1" : "0";
  }

  function resetPhysicalVisuals() {
    var panel = document.getElementById("bankPanel");
    var mappingPath = document.getElementById("mappingPath");
    var vectorPath = document.getElementById("vectorToGbPath");
    var runtime = document.getElementById("runtimePanel");
    if (panel) panel.classList.remove("is-revealed");
    if (mappingPath) mappingPath.classList.remove("is-active");
    if (vectorPath) vectorPath.classList.remove("is-active");
    if (runtime) runtime.classList.remove("is-active", "is-streaming");
    document.querySelectorAll("#physicalViz [data-active-bank='true']").forEach(function (cell) {
      cell.classList.remove("is-active");
      cell.style.animationDelay = "0ms";
    });
    document.querySelectorAll("#physicalViz [data-packet], #physicalViz [data-vector-packet]").forEach(function (node) {
      setAnimatedRect(node, 0);
    });
    document.querySelectorAll("#physicalViz [data-gb-slot]").forEach(function (gbSlot) {
      gbSlot.classList.remove("is-loading", "is-loaded");
    });
    renderWaveState(-1, 0);
    setPhase(1);
  }

  function revealBankPanel() {
    var panel = document.getElementById("bankPanel");
    var path = document.getElementById("mappingPath");
    if (panel) panel.classList.add("is-revealed");
    if (path) path.classList.add("is-active");
  }

  function activateBankCells(animate) {
    document.querySelectorAll("#physicalViz [data-active-bank='true']").forEach(function (cell, index) {
      cell.style.animationDelay = (animate && !reducedMotion() ? 130 + (index % 32) * 5 + Math.floor(index / 32) * 6 : 0) + "ms";
      cell.classList.add("is-active");
    });
  }

  function finishBankPlacement() {
    activateBankCells(false);
    document.querySelectorAll("#physicalViz [data-packet]").forEach(function (node) { setAnimatedRect(node, 1); });
    setPhase(2);
  }

  function animateBankPlacement(token, callback) {
    activateBankCells(true);
    var packets = Array.prototype.slice.call(document.querySelectorAll("#physicalViz [data-packet]"));
    var duration = reducedMotion() ? 1 : 700;
    var stagger = reducedMotion() ? 0 : Math.min(18, 180 / Math.max(packets.length, 1));
    var start = performance.now() + (reducedMotion() ? 0 : 120);
    function frame(now) {
      if (token !== state.animationToken) return;
      var complete = true;
      packets.forEach(function (node, index) {
        var t = Math.max(0, Math.min(1, (now - start - index * stagger) / duration));
        if (t < 1) complete = false;
        setAnimatedRect(node, t);
      });
      if (!complete) requestAnimationFrame(frame);
      else if (callback) callback();
    }
    requestAnimationFrame(frame);
  }

  function finishVectorToGb() {
    var vectors = document.querySelectorAll("#physicalViz [data-vector-packet]");
    var path = document.getElementById("vectorToGbPath");
    var runtime = document.getElementById("runtimePanel");
    if (runtime) runtime.classList.add("is-active");
    if (path) path.classList.add("is-active");
    vectors.forEach(function (vector) { setAnimatedRect(vector, 1); });
    document.querySelectorAll("#physicalViz [data-gb-slot]").forEach(function (slot) {
      slot.classList.remove("is-loading");
      slot.classList.add("is-loaded");
    });
    setPhase(3);
  }

  function animateVectorToGb(token, callback) {
    var vectors = Array.prototype.slice.call(document.querySelectorAll("#physicalViz [data-vector-packet]"));
    var path = document.getElementById("vectorToGbPath");
    if (!vectors.length) {
      if (callback) callback();
      return;
    }
    if (path) path.classList.add("is-active");
    document.querySelectorAll("#physicalViz [data-gb-slot]").forEach(function (slot) { slot.classList.add("is-loading"); });
    var duration = reducedMotion() ? 1 : 680;
    var start = performance.now() + (reducedMotion() ? 0 : 90);
    function frame(now) {
      if (token !== state.animationToken) return;
      var t = Math.max(0, Math.min(1, (now - start) / duration));
      vectors.forEach(function (vector) { setAnimatedRect(vector, t); });
      if (t < 1) requestAnimationFrame(frame);
      else {
        finishVectorToGb();
        if (callback) callback();
      }
    }
    requestAnimationFrame(frame);
  }

  function startSaStream() {
    var runtime = document.getElementById("runtimePanel");
    if (!runtime) return;
    runtime.classList.remove("is-streaming");
    /* Force a style flush so a clicked phase reliably restarts the wave. */
    runtime.getBoundingClientRect();
    runtime.classList.add("is-active", "is-streaming");
  }

  function runPhaseFour(token) {
    setPhase(4);
    var wavePanel = document.getElementById("wavePanel");
    if (!wavePanel) {
      startSaStream();
      document.getElementById("physicalHint").textContent = "Phase 4/4 · " + saLabel() + " is streaming · click any phase to replay";
      return;
    }
    var waveCount = Number(wavePanel.getAttribute("data-wave-count"));
    var waveK = Number(wavePanel.getAttribute("data-wave-k"));
    var waveN = Number(wavePanel.getAttribute("data-wave-n"));
    var parallelKParts = Number(wavePanel.getAttribute("data-parallel-k-parts"));
    var svPartitionMode = wavePanel.getAttribute("data-sv-partition-mode");
    var banks = Number(document.querySelector("#physicalViz svg").getAttribute("data-bank-count"));
    if (reducedMotion()) {
      startSaStream();
      renderWaveState(-1, waveCount);
      document.getElementById("physicalHint").textContent = "Phase 4/4 · C0 complete after " + waveCount + " wave(s)" + (state.mapping === "reduction_split" ? " · local Σ C0…C31" : "");
      return;
    }
    var wave = 0;
    var waveDuration = Math.max(4, Math.min(650, 3200 / Math.max(waveCount, 1)));
    function streamNextWave() {
      if (token !== state.animationToken) return;
      renderWaveState(wave, wave);
      startSaStream();
      var waveWork = svPartitionMode === "k_chunks"
        ? parallelKParts + " K-parts × (" + number(waveK) + " × " + number(waveN) + ")"
        : svPartitionMode === "output_slices"
          ? number(waveK) + " K × " + number(waveN) + " N over two N-halves"
          : number(waveK) + " × " + number(waveN);
      document.getElementById("physicalHint").textContent = "Phase 4/4 · C0 wave " + (wave + 1) + "/" + waveCount + " · " + waveWork + " across " + banks + " banks";
      schedule(token, waveDuration, function () {
        wave += 1;
        renderWaveState(-1, wave);
        if (wave < waveCount) {
          streamNextWave();
        } else {
          document.getElementById("physicalHint").textContent = state.mapping === "reduction_split" ? "Phase 4/4 · C0 complete · C0…C31 partials reduce into the output" : "Phase 4/4 · C0 complete · companion channels/groups follow the same schedule";
        }
      });
    }
    streamNextWave();
  }

  function playPhysicalStage(phase) {
    var token = ++state.animationToken;
    resetPhysicalVisuals();
    if (phase === 1) {
      document.getElementById("physicalHint").textContent = "Phase 1/4 · select the highlighted [K rows × N columns] tile";
      return;
    }
    revealBankPanel();
    if (phase === 2) {
      document.getElementById("physicalHint").textContent = "Phase 2/4 · replaying tile placement into banks";
      animateBankPlacement(token, function () { setPhase(2); });
      return;
    }
    finishBankPlacement();
    document.getElementById("runtimePanel").classList.add("is-active");
    if (phase === 3) {
      document.getElementById("physicalHint").textContent = "Phase 3/4 · replaying vector K-slice load into GB";
      animateVectorToGb(token, function () { setPhase(3); });
      return;
    }
    finishVectorToGb();
    runPhaseFour(token);
  }

  function schedule(token, delay, callback) {
    window.setTimeout(function () {
      if (token === state.animationToken) callback();
    }, delay);
  }

  function setPhase(phase) {
    document.querySelectorAll("#physicalViz [data-phase]").forEach(function (node) {
      var value = Number(node.getAttribute("data-phase"));
      node.classList.toggle("is-active", value <= phase);
    });
  }

  function renderPhysicalFacts(kernel, layout) {
    var facts;
    if (state.mapping === "reduction_split") {
      facts = [
        ["Mapping policy", "full N / split K", "Each of 32 channels owns the complete output-column range and 1/32 of the reduction axis."],
        ["Selected C0 tile", number(layout.channelTileK) + " × " + number(layout.channelTileN), "K rows × N columns after the 32-way channel split."],
        ["C0 wave schedule", layout.outputWaves + " × " + layout.kChunks + " = " + number(layout.totalWaves), layout.outputWaves + " output wave(s) × " + layout.kChunks + " physical K wave(s); one wave is " + waveShapeLabel(layout, layout.channelTileK, layout.channelTileN) + "."],
        [kernel.split === "row" ? "Two reductions" : "Local reduction", kernel.split === "row" ? "Σ channels → TP" : "Σ C0…C31", kernel.split === "row" ? "First combine 32 channel partials locally, then TP All-Reduce combines device partials." : "The 32 synchronized channel partials combine inside the device; no TP collective is added.", kernel.split === "row" ? "collective" : "local"]
      ];
      if (kernel.kind === "qk" || kernel.kind === "sv") {
        facts.splice(1, 0, ["Attention rounds", layout.localKvHeads + " KV × 4 batch × 2 GQA", "Each SA launch packs four GQA heads from one batch. The four batch caches stay independent; eight GQA heads take two waves."]);
      }
      if (kernel.kind === "sv") {
        facts.splice(2, 0, ["SV SA partition", svPartitionSummary(layout), svPartitionExplanation(layout)]);
      }
    } else if (kernel.kind === "qk") {
      facts = [
        ["Attention rounds", "4 batch × 2 GQA", "Each cube is a separate batch cache. One SA launch uses four GQA heads from one batch; eight heads take two waves."],
        ["Physical allocation", number(layout.banksPerHead) + " banks/head", layout.channelsPerHead + " channels per local KV head."],
        ["Selected channel tile", "128 × " + number(layout.contextsPerChannel), "C0 is sequence-striped across " + layout.banksPerChannel + " physical banks."],
        ["C0 wave schedule", layout.outputWaves + " × " + layout.kChunks + " = " + number(layout.totalWaves), layout.outputWaves + " output wave(s) × " + layout.kChunks + " physical K wave(s); one wave is " + waveShapeLabel(layout, layout.channelTileK, layout.channelTileN) + ".", "local"]
      ];
    } else if (kernel.kind === "sv") {
      facts = [
        ["Attention rounds", "4 batch × 2 GQA", "Each cube is a separate batch cache. One SA launch uses four GQA score rows from one batch; eight heads take two waves."],
        ["SV SA partition", svPartitionSummary(layout), svPartitionExplanation(layout)],
        ["128-dim group", layout.physicalGroupChannels + " channel(s)", dramLabel(state.dram) + " has " + layout.banksPerChannel + " banks/channel."],
        ["C0 wave schedule", layout.outputWaves + " × " + layout.kChunks + " = " + number(layout.totalWaves), "GB stages " + gbElementsPerRow(kernel, layout) + " context values/K-part/row; companion context groups reduce locally.", "local"]
      ];
    } else {
      facts = [
        ["Device geometry", "32 × " + layout.banksPerChannel + " banks", number(layout.totalBanks) + " banks · " + layout.burstLength + " " + layout.precision + " outputs/bank word."],
        ["Channel grouping", layout.channelsPerGroup + " ch/group", layout.reductionGroups + " independent reduction group(s)."],
        ["Selected C0 tile", number(layout.channelTileK) + " × " + number(layout.channelTileN), "K rows × N columns assigned to the representative channel."],
        ["C0 wave schedule", layout.outputWaves + " × " + layout.kChunks + " = " + number(layout.totalWaves), "GB stages 4 × " + gbElementsPerRow(kernel, layout) + " " + layout.precision + " values/row-wave, then streams through " + saLabel() + ".", kernel.split === "row" ? "collective" : "local"]
      ];
    }
    document.getElementById("physicalFacts").innerHTML = facts.map(factHtml).join("");
  }

  function resetPhysicalMapping() {
    state.animationToken += 1;
    renderLevel3();
  }

  var SVG_EXPORT_STYLE_PROPERTIES = [
    "fill", "fill-opacity", "fill-rule",
    "stroke", "stroke-opacity", "stroke-width", "stroke-dasharray", "stroke-dashoffset",
    "stroke-linecap", "stroke-linejoin", "stroke-miterlimit",
    "opacity", "color", "visibility",
    "font-family", "font-size", "font-style", "font-variant", "font-weight",
    "letter-spacing", "text-anchor", "dominant-baseline", "alignment-baseline",
    "paint-order", "shape-rendering", "vector-effect",
    "display", "overflow", "transform", "transform-box", "transform-origin",
    "marker-start", "marker-mid", "marker-end",
    "clip-path", "mask", "stop-color", "stop-opacity"
  ];

  function inlineSvgComputedStyles(source, clone) {
    var sourceNodes = [source].concat(Array.prototype.slice.call(source.querySelectorAll("*")));
    var cloneNodes = [clone].concat(Array.prototype.slice.call(clone.querySelectorAll("*")));
    sourceNodes.forEach(function (sourceNode, index) {
      var cloneNode = cloneNodes[index];
      if (!cloneNode || sourceNode.nodeType !== 1) return;
      var computed = window.getComputedStyle(sourceNode);
      cloneNode.removeAttribute("style");
      SVG_EXPORT_STYLE_PROPERTIES.forEach(function (property) {
        var value = computed.getPropertyValue(property);
        if (value) cloneNode.style.setProperty(property, value);
      });
      cloneNode.style.setProperty("animation", "none");
      cloneNode.style.setProperty("transition", "none");
      cloneNode.style.setProperty("filter", "none");
      cloneNode.removeAttribute("class");
      cloneNode.removeAttribute("tabindex");
      cloneNode.removeAttribute("role");
      Array.prototype.slice.call(cloneNode.attributes).forEach(function (attribute) {
        if (attribute.name.indexOf("data-") === 0) cloneNode.removeAttribute(attribute.name);
      });
    });
  }

  function addSvgExportBackground(clone) {
    var viewBox = String(clone.getAttribute("viewBox") || "0 0 1200 560").trim().split(/\s+/).map(Number);
    if (viewBox.length !== 4 || viewBox.some(function (value) { return !Number.isFinite(value); })) {
      viewBox = [0, 0, 1200, 560];
    }
    clone.setAttribute("width", viewBox[2]);
    clone.setAttribute("height", viewBox[3]);
    var background = svgEl("rect", {
      x: viewBox[0],
      y: viewBox[1],
      width: viewBox[2],
      height: viewBox[3],
      fill: "#ffffff",
      stroke: "none"
    });
    var defs = clone.querySelector("defs");
    if (defs && defs.nextSibling) clone.insertBefore(background, defs.nextSibling);
    else clone.insertBefore(background, clone.firstChild);
  }

  function prepareSvgExport(source) {
    var clone = source.cloneNode(true);
    clone.setAttribute("version", "1.1");
    inlineSvgComputedStyles(source, clone);
    addSvgExportBackground(clone);
    return clone;
  }

  function exportSvg() {
    var selector = state.level === 3 ? "#physicalViz svg" : "#partitionViz svg";
    var source = document.querySelector(selector);
    if (!source) return;
    var clone = prepareSvgExport(source);
    var serialized = new XMLSerializer().serializeToString(clone);
    var blob = new Blob([serialized], { type: "image/svg+xml;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "llama31-70b-" + state.kernel + "-tp" + state.tp + "-" + contextLabel(state.context).toLowerCase() + "-" + state.dram + "-" + state.precision + "-level" + state.level + ".svg";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    showToast("Exported the current SVG view.");
  }

  function showToast(message) {
    var toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.add("is-visible");
    if (state.toastTimer) clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(function () { toast.classList.remove("is-visible"); }, 2200);
  }

  function bindEvents() {
    document.querySelectorAll(".segmented-control").forEach(function (control) {
      var key = control.getAttribute("data-control");
      control.querySelectorAll("button").forEach(function (button) {
        button.addEventListener("click", function () {
          var raw = button.getAttribute("data-value");
          var value = key === "dram" || key === "precision" || key === "mapping" ? raw : Number(raw);
          if (state[key] === value) return;
          state[key] = value;
          if (key === "precision" && value === "fp8") state.dram = "lpddr4x";
          if (key === "tp" && state.device >= state.tp) state.device = 0;
          state.animationToken += 1;
          renderAll();
          if (key === "context" && state.kernel && kernels[state.kernel].kind === "dense") {
            showToast("Context changes KV-cache kernels only; this weight-matrix shape is unchanged.");
          }
          if (key === "precision" && value === "fp8") {
            showToast("FP8 uses the validated LPDDR4X geometry: one 256-bit word = 32 elements.");
          }
          if (key === "mapping") {
            showToast(value === "reduction_split" ? "Reduction Split: every channel owns full N and one K/32 slice." : "Channel Group: channels split N while channel groups partition K.");
          }
        });
      });
    });

    document.querySelectorAll("[data-kernel]").forEach(function (button) {
      button.addEventListener("click", function () { selectKernel(button.getAttribute("data-kernel")); });
    });
    document.getElementById("backButton").addEventListener("click", goBack);
    document.getElementById("replayPartition").addEventListener("click", function () { renderLevel2(); });
    document.getElementById("replayMapping").addEventListener("click", resetPhysicalMapping);
    document.getElementById("exportButton").addEventListener("click", exportSvg);
  }

  readUrl();
  bindEvents();
  updateControls();
  updateSummary();
  setLevel(state.level);
  if (state.level === 2) renderLevel2();
  if (state.level === 3) renderLevel3();
})();
