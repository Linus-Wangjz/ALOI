(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
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
      return "q [1 × 128] × K_headᵀ [128 × " + contextLabel(state.context) + "] → score [1 × " + contextLabel(state.context) + "]";
    }
    if (kernel.kind === "sv") {
      return "score [1 × " + contextLabel(state.context) + "] × V_head [" + contextLabel(state.context) + " × 128] → o [1 × 128]";
    }
    return "x [1 × " + number(shape.k) + "] × W_local [" + number(shape.k) + " × " + number(shape.n) + "] → y_local [1 × " + number(shape.n) + "]";
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
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
    });
  }

  function updateSummary() {
    var bpc = bankCount();
    var html = [
      ["TP group", state.tp + " device" + (state.tp > 1 ? "s" : "")],
      ["Context", contextLabel(state.context)],
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
      parts.push("Bank mapping");
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
    var kernel = params.get("kernel");
    var level = Number(params.get("level"));
    var device = Number(params.get("device"));
    if ([1, 2, 4, 8].indexOf(tp) >= 0) state.tp = tp;
    if ([32768, 131072].indexOf(context) >= 0) state.context = context;
    if (["gddr6", "lpddr4x"].indexOf(dram) >= 0) state.dram = dram;
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
    document.getElementById("partitionEquation").textContent = kernel.equation;
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
    append(svg, "text", { x: 235, y: 269, "class": "svg-mono", "text-anchor": "middle" }, "[B, 8192]");
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
        ["Per-device operand", localShapeLabel(kernel), kernel.localOutput(state)],
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
    var totalBanks = 32 * bpc;
    var capacity = totalBanks * 16;
    var local = localShape(kernel);
    var result = {
      banksPerChannel: bpc,
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
      result.outputTiles = Math.ceil(local.n / capacity);
      result.activeChannels = result.outputTiles > 1
        ? 32
        : Math.ceil(local.n / (bpc * 16));
      if (kernel.short === "Wq" || kernel.short === "[Wk | Wv]") {
        var tileOutputs = Math.min(local.n, capacity);
        result.channelsPerGroup = Math.ceil(tileOutputs / (bpc * 16));
        result.reductionGroups = Math.floor(32 / result.channelsPerGroup);
        result.reductionSlice = Math.ceil(local.k / result.reductionGroups);
        result.activeChannels = result.channelsPerGroup * result.reductionGroups;
      }
      result.representativeChannels = [];
      for (var denseChannel = 0; denseChannel < Math.min(result.channelsPerGroup, result.activeChannels); denseChannel += 1) {
        result.representativeChannels.push(denseChannel);
      }
      return result;
    }

    var localKvHeads = 8 / state.tp;
    result.localKvHeads = localKvHeads;
    result.banksPerHead = totalBanks / localKvHeads;
    result.channelsPerHead = 32 / localKvHeads;
    if (kernel.kind === "qk") {
      result.contextsPerBank = Math.ceil(state.context / result.banksPerHead);
      result.sequenceWaves = Math.ceil(result.contextsPerBank / 16);
      result.queryWaves = 2;
      result.representativeChannels = [];
      for (var qkChannel = 0; qkChannel < result.channelsPerHead; qkChannel += 1) result.representativeChannels.push(qkChannel);
      return result;
    }

    var physicalGroupChannels = 128 / (bpc * 8);
    result.physicalGroupChannels = physicalGroupChannels;
    result.svMode = localKvHeads >= 2 ? "paired KV heads" : "paired query groups";
    if (localKvHeads >= 2) {
      var workItems = localKvHeads / 2;
      var channelsPerWorkItem = 32 / workItems;
      result.contextGroups = channelsPerWorkItem / physicalGroupChannels;
      result.queryWaves = 2;
    } else {
      result.contextGroups = 32 / physicalGroupChannels;
      result.queryWaves = 1;
    }
    result.contextsPerHalf = Math.ceil(state.context / result.contextGroups);
    result.representativeChannels = [];
    for (var c = 0; c < physicalGroupChannels; c += 1) result.representativeChannels.push(c);
    return result;
  }

  function renderLevel3() {
    var kernel = kernels[state.kernel];
    if (!kernel || kernel.kind === "local") return;
    var layout = physicalLayout(kernel);
    document.getElementById("level3-title").textContent = kernel.short + " · Device " + state.device;
    document.getElementById("level3-subtitle").textContent = "Follow one GEMV from the transposed TP-local tensor through " + dramLabel(state.dram) + " channel groups, bank rows, GB and SA4×16.";
    document.getElementById("physicalEquation").textContent = gemvEquation(kernel);
    document.getElementById("physicalHint").textContent = "Click the tile or one of the four phases";
    renderPhysicalSvg(kernel, layout);
    renderPhysicalFacts(kernel, layout);
  }

  function renderPhysicalSvg(kernel, layout) {
    var host = document.getElementById("physicalViz");
    host.setAttribute("aria-label", kernel.title + " physical mapping inside Device " + state.device);
    var svg = svgRoot(host, "0 0 1360 820");
    addArrowDefs(svg);
    var color = DEVICE_COLORS[state.device];
    var matrixShape = physicalMatrixShape(kernel);
    /* The screen uses the GEMV point of view: K runs left-to-right and N top-to-bottom. */
    var frame = fitRect(matrixShape.n, matrixShape.k, { x: 70, y: 220, w: 445, h: 270, minW: 180, minH: 108 });

    append(svg, "text", { x: 54, y: 49, "class": "svg-kicker" }, "TP-LOCAL OPERAND");
    append(svg, "text", { x: 54, y: 75, "class": "svg-title" }, "Device " + state.device + " · " + localShapeLabel(kernel));
    append(svg, "text", { x: 54, y: 94, "class": "svg-subtitle" }, physicalOperandSubtitle(kernel, layout));

    append(svg, "text", { x: 54, y: 126, "class": "svg-kicker" }, "GEMV RUNTIME VECTOR");
    append(svg, "text", { x: 54, y: 143, "class": "svg-subtitle" }, "Horizontal x[K]; highlighted K-slice is staged in the Global Buffer");

    append(svg, "rect", { x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 8, "class": "matrix-frame" });
    drawLocalMatrixGrid(svg, frame, kernel, layout, color);

    append(svg, "text", { x: frame.x + frame.w / 2, y: frame.y + frame.h + 27, "class": "svg-small", "text-anchor": "middle" }, "K · " + number(matrixShape.k) + " (reduction / vector axis)");
    append(svg, "text", {
      x: frame.x - 23,
      y: frame.y + frame.h / 2,
      "class": "svg-small",
      "text-anchor": "middle",
      transform: "rotate(-90 " + (frame.x - 23) + " " + (frame.y + frame.h / 2) + ")"
    }, "N · " + number(matrixShape.n) + " (output axis)");
    append(svg, "text", { x: frame.x + 10, y: frame.y + 17, "class": "matrix-view-label" }, "display transpose · rows N, columns K");

    var selected = selectedTile(frame, kernel, layout);
    var tileGroup = append(svg, "g", {
      "class": "matrix-tile-clickable",
      role: "button",
      tabindex: "0",
      "aria-label": "Map selected matrix tile to DRAM banks"
    });
    append(tileGroup, "rect", {
      x: selected.x,
      y: selected.y,
      width: selected.w,
      height: selected.h,
      rx: 4,
      "class": "selected-tile"
    });
    var compactTile = selected.w < 88 || selected.h < 62;
    var tileLabelX = compactTile ? selected.x + selected.w + 8 : selected.x + selected.w / 2;
    var tileLabelY = compactTile ? selected.y + Math.min(14, selected.h / 2) : selected.y + selected.h / 2 - 4;
    textLines(tileGroup, tileLabelX, tileLabelY, selected.lines, "svg-mono", 13, compactTile ? "start" : "middle");
    append(tileGroup, "text", {
      x: compactTile ? tileLabelX : selected.x + selected.w / 2,
      y: compactTile ? tileLabelY + 29 : selected.y + selected.h / 2 + 27,
      fill: "#1744a5",
      "font-size": 8,
      "font-weight": 760,
      "text-anchor": compactTile ? "start" : "middle"
    }, "CLICK · PLAY 1→4");
    tileGroup.addEventListener("click", revealPhysicalMapping);
    tileGroup.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") revealPhysicalMapping();
    });

    append(svg, "path", {
      d: "M " + (selected.x + selected.w) + " " + (selected.y + selected.h / 2) +
        " C 535 " + (selected.y + selected.h / 2) + ", 555 245, 650 245",
      "class": "mapping-path",
      id: "mappingPath"
    });

    var bankPanel = append(svg, "g", { id: "bankPanel", "class": "bank-panel-svg" });
    append(bankPanel, "rect", { x: 590, y: 35, width: 720, height: 650, rx: 16, fill: "#fbfcfe", stroke: "#cdd6e3" });
    append(bankPanel, "text", { x: 615, y: 65, "class": "svg-kicker" }, "PHYSICAL DEVICE");
    append(bankPanel, "text", { x: 615, y: 88, "class": "svg-title" }, dramLabel(state.dram) + " · 32 channels × " + layout.banksPerChannel + " banks");
    append(bankPanel, "text", { x: 615, y: 105, "class": "svg-subtitle" }, bankPanelSubtitle(kernel, layout));

    var bankGeometry = drawBankGrid(bankPanel, kernel, layout, color);
    var runtimeGeometry = drawRuntimePanel(bankPanel, kernel, layout);
    drawGemvVector(svg, frame, kernel, layout, color, runtimeGeometry.gb);
    drawPackets(svg, selected, bankGeometry, layout, kernel, color);
    drawExecutionPhases(svg);
    wireChannelGroupHighlight(svg);

    append(svg, "text", { x: 54, y: 565, "class": "svg-kicker" }, "SELECTED TILE RULE");
    textLines(svg, 54, 589, selectedRuleLines(kernel, layout), "svg-subtitle", 17);
    append(svg, "text", { x: 54, y: 720, "class": "svg-kicker" }, "INTERACTIVE PIPELINE · CLICK ANY STAGE TO REPLAY IT");

    svg.setAttribute("data-bank-count", layout.banksPerChannel);
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

  function drawLocalMatrixGrid(svg, frame, kernel, layout, color) {
    var groupCount = kernel.kind === "sv" ? Math.min(layout.contextGroups, 32) : kernel.kind === "qk" ? 1 : Math.min(layout.reductionGroups, 32);
    var channelsInGroup = kernel.kind === "qk" ? layout.channelsPerHead : kernel.kind === "sv" ? layout.physicalGroupChannels : layout.channelsPerGroup;
    var groupWidth = frame.w / Math.max(groupCount, 1);

    for (var group = 0; group < groupCount; group += 1) {
      var channelStart = kernel.kind === "qk" ? 0 : kernel.kind === "sv" ? group * layout.physicalGroupChannels : group * layout.channelsPerGroup;
      channelStart = channelStart % 32;
      var groupColor = channelColor(kernel, layout, channelStart, color);
      var gx = frame.x + group * groupWidth;
      append(svg, "rect", {
        x: gx,
        y: frame.y,
        width: groupWidth,
        height: frame.h,
        fill: groupColor.fill,
        opacity: group === 0 ? 0.68 : 0.42,
        "class": "matrix-channel-group-fill"
      });

      var visibleChannels = Math.max(1, Math.min(channelsInGroup, 32));
      for (var slice = 0; slice < visibleChannels; slice += 1) {
        var sy = frame.y + slice * frame.h / visibleChannels;
        append(svg, "rect", {
          x: gx,
          y: sy,
          width: groupWidth,
          height: frame.h / visibleChannels,
          fill: groupColor.strong,
          opacity: 0.045 + (slice % 3) * 0.035,
          "class": "matrix-channel-slice",
          "data-channel-slice": (channelStart + slice) % 32,
          "data-channel-group": group
        });
        if (slice > 0) {
          append(svg, "line", { x1: gx, y1: sy, x2: gx + groupWidth, y2: sy, "class": "matrix-grid-line fine" });
        }
        if (group === 0 && groupWidth >= 40 && frame.h / visibleChannels >= 7) {
          append(svg, "text", {
            x: gx + 4,
            y: sy + Math.min(frame.h / visibleChannels - 1.5, 7),
            fill: groupColor.text,
            "font-size": 5.8,
            "font-weight": 720,
            "data-channel-label": (channelStart + slice) % 32
          }, "C" + ((channelStart + slice) % 32));
        }
      }

      append(svg, "rect", {
        x: gx,
        y: frame.y,
        width: groupWidth,
        height: frame.h,
        rx: groupCount === 1 ? 8 : 1,
        fill: "none",
        stroke: groupColor.strong,
        "stroke-width": group === 0 ? 2.8 : 1.5,
        "class": "channel-group-outline" + (group === 0 ? " is-highlighted" : ""),
        "data-channel-group-outline": group
      });

      if (groupWidth >= 42 || group === 0 || group === groupCount - 1) {
        var lastChannel = (channelStart + channelsInGroup - 1) % 32;
        append(svg, "text", {
          x: gx + groupWidth / 2,
          y: frame.y + frame.h - 8,
          fill: groupColor.text,
          "font-size": groupWidth < 42 ? 6.5 : 8,
          "font-weight": 790,
          "text-anchor": "middle",
          "class": "channel-group-label"
        }, "G" + group + " · C" + channelStart + (channelsInGroup > 1 ? "–C" + lastChannel : ""));
      }
    }

    append(svg, "rect", { x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 8, fill: "none", stroke: color.strong, "stroke-width": 1.7 });
    append(svg, "line", { x1: frame.x, y1: frame.y - 15, x2: frame.x + 22, y2: frame.y - 15, stroke: color.strong, "stroke-width": 3 });
    append(svg, "text", { x: frame.x + 28, y: frame.y - 12, "class": "svg-small" }, "same colored outline = one channel group");
  }

  function vectorGroupCount(kernel, layout) {
    if (kernel.kind === "qk") return 1;
    if (kernel.kind === "sv") return Math.min(layout.contextGroups, 32);
    return Math.min(layout.reductionGroups, 32);
  }

  function drawGemvVector(svg, frame, kernel, layout, color, gbTarget) {
    var y = 166;
    var h = 22;
    var groups = Math.max(1, vectorGroupCount(kernel, layout));
    var segmentW = frame.w / groups;
    append(svg, "text", { x: frame.x - 9, y: y + 15, "class": "svg-mono", "text-anchor": "end" }, "x");
    for (var group = 0; group < groups; group += 1) {
      var channel = kernel.kind === "sv" ? group * layout.physicalGroupChannels : kernel.kind === "qk" ? 0 : group * layout.channelsPerGroup;
      channel %= 32;
      var segmentColor = channelColor(kernel, layout, channel, color);
      append(svg, "rect", {
        x: frame.x + group * segmentW,
        y: y,
        width: segmentW,
        height: h,
        rx: groups === 1 ? 4 : 1,
        fill: segmentColor.fill,
        stroke: segmentColor.strong,
        "stroke-width": group === 0 ? 2 : 0.8,
        "data-vector-slice": group,
        "data-channel-group": group
      });
      if ((segmentW > 34 || group === 0 || group === groups - 1) && groups > 1) {
        append(svg, "text", {
          x: frame.x + (group + 0.5) * segmentW,
          y: y + 15,
          fill: segmentColor.text,
          "font-size": segmentW < 34 ? 6.5 : 8,
          "font-weight": 760,
          "text-anchor": "middle"
        }, "K" + group);
      }
    }
    append(svg, "text", { x: frame.x + frame.w + 9, y: y + 15, "class": "svg-mono" }, "[1 × " + number(physicalMatrixShape(kernel).k) + "]");
    append(svg, "path", {
      d: "M " + (frame.x + frame.w / 2) + " " + (y + h + 3) + " V " + (frame.y - 6),
      stroke: "#91a0b5",
      "stroke-width": 1.2,
      "marker-end": "url(#arrowhead)"
    });

    var groupK = kernel.kind === "qk" ? 128 : kernel.kind === "sv" ? layout.contextsPerHalf : layout.reductionSlice;
    var chunkK = kernel.kind === "qk" ? 128 : kernel.kind === "sv" ? Math.min(128, groupK) : Math.min(256, groupK);
    var selectedW = kernel.kind === "qk" ? frame.w : Math.max(5, segmentW * chunkK / Math.max(groupK, 1));
    append(svg, "rect", {
      x: frame.x,
      y: y - 2,
      width: selectedW,
      height: h + 4,
      rx: 3,
      "class": "selected-vector-slice",
      "data-selected-vector-k": chunkK
    });
    append(svg, "text", { x: frame.x, y: y - 7, fill: "#b66a1d", "font-size": 7.5, "font-weight": 790 }, "GB slice · " + number(chunkK) + " K elements");
    var vectorPacket = append(svg, "rect", {
      x: frame.x,
      y: y,
      width: selectedW,
      height: h,
      rx: 3,
      "class": "vector-packet",
      "data-vector-packet": "true",
      "data-sx": frame.x,
      "data-sy": y,
      "data-sw": selectedW,
      "data-sh": h,
      "data-tx": gbTarget.x + 12,
      "data-ty": gbTarget.y + gbTarget.h / 2 - 5,
      "data-tw": Math.max(25, gbTarget.w - 24),
      "data-th": 10
    });
    vectorPacket.style.fill = color.strong;
    append(svg, "path", {
      d: "M " + (frame.x + selectedW / 2) + " " + (y - 4) + " C 520 120, 570 575, " + (gbTarget.x - 8) + " " + (gbTarget.y + gbTarget.h / 2),
      "class": "vector-to-gb-path",
      id: "vectorToGbPath"
    });
  }

  function selectedTile(frame, kernel, layout) {
    if (kernel.kind === "qk") {
      return {
        x: frame.x + frame.w * 0.08,
        y: frame.y + frame.h * 0.06,
        w: frame.w * 0.84,
        h: Math.max(55, frame.h * 0.19),
        lines: ["one QK wave", "128 × 16 context"]
      };
    }
    if (kernel.kind === "sv") {
      return {
        x: frame.x + frame.w * 0.02,
        y: frame.y + frame.h * 0.06,
        w: Math.max(8, frame.w / Math.max(layout.contextGroups, 1)),
        h: frame.h * 0.88,
        lines: ["one context group", number(layout.contextsPerHalf) + " × 128"]
      };
    }
    var tileFraction = Math.min(1, layout.outputCapacity / Math.max(layout.localN, 1));
    var groupFraction = 1 / Math.max(layout.reductionGroups, 1);
    return {
      x: frame.x,
      y: frame.y,
      w: Math.max(8, frame.w * groupFraction),
      h: Math.max(8, frame.h * tileFraction),
      lines: [
        "matrix tile",
        number(layout.reductionSlice) + " × " + number(Math.min(layout.localN, layout.outputCapacity))
      ]
    };
  }

  function bankPanelSubtitle(kernel, layout) {
    if (kernel.kind === "qk") {
      return number(layout.banksPerHead) + " banks/KV head · sequence-striped K cache";
    }
    if (kernel.kind === "sv") {
      return layout.physicalGroupChannels + " channel(s)/128-dim group · " + layout.svMode;
    }
    return layout.channelsPerGroup + " channel(s)/group · " + layout.reductionGroups + " reduction group(s)";
  }

  function activeBank(kernel, layout, channel) {
    if (kernel.kind === "qk") return true;
    if (kernel.kind === "sv") return true;
    return channel < layout.activeChannels;
  }

  function channelColor(kernel, layout, channel, deviceColor) {
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

    append(parent, "text", { x: 615, y: y - 13, "class": "svg-small" }, "CHANNEL / GROUP");
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
    if (kernel.kind === "qk") {
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

  function drawRuntimePanel(parent, kernel, layout) {
    var runtime = append(parent, "g", { id: "runtimePanel", "class": "runtime-panel" });
    var gb = { x: 650, y: 548, w: 180, h: 72 };
    var sa = { x: 935, y: 548, w: 148, h: 72 };
    append(runtime, "line", { x1: 615, y1: 508, x2: 1285, y2: 508, stroke: "#d5dde8" });
    append(runtime, "text", { x: 615, y: 531, "class": "svg-kicker" }, "RUNTIME DATAFLOW · WEIGHT ROWS ↓  /  VECTOR SLICE →");

    append(runtime, "rect", { x: gb.x, y: gb.y, width: gb.w, height: gb.h, rx: 10, "class": "gb-box" });
    append(runtime, "text", { x: gb.x + gb.w / 2, y: gb.y + 18, fill: "#7d4b14", "font-size": 10, "font-weight": 780, "text-anchor": "middle" }, "GLOBAL BUFFER (GB)");
    append(runtime, "text", { x: gb.x + gb.w / 2, y: gb.y + 33, fill: "#a46928", "font-size": 8, "text-anchor": "middle" }, runtimeOperand(kernel));
    append(runtime, "rect", { x: gb.x + 12, y: gb.y + 43, width: gb.w - 24, height: 12, rx: 3, "class": "gb-vector-slot", "data-gb-slot": "true" });
    append(runtime, "text", { x: gb.x + gb.w / 2, y: gb.y + 66, fill: "#9a6427", "font-size": 7.5, "text-anchor": "middle" }, gbCapacityLabel(kernel));

    append(runtime, "path", { d: "M " + (gb.x + gb.w) + " " + (gb.y + gb.h / 2) + " H " + (sa.x - 10), stroke: "#8d9caf", "stroke-width": 1.5, "marker-end": "url(#arrowhead)", "class": "runtime-arrow" });
    append(runtime, "path", { d: "M 1115 490 C 1115 525, 1035 523, 1035 " + (sa.y - 8), stroke: "#8d9caf", "stroke-width": 1.5, fill: "none", "marker-end": "url(#arrowhead)", "class": "runtime-arrow" });
    append(runtime, "text", { x: 1120, y: 523, fill: "#748196", "font-size": 7.5 }, "bank-row stream");

    append(runtime, "rect", { x: sa.x, y: sa.y, width: sa.w, height: sa.h, rx: 10, "class": "sa-box" });
    append(runtime, "text", { x: sa.x + 11, y: sa.y + 17, fill: "#08675f", "font-size": 10, "font-weight": 790 }, "SA 4×16");
    append(runtime, "text", { x: sa.x + sa.w - 10, y: sa.y + 17, fill: "#328f86", "font-size": 7.5, "text-anchor": "end" }, "MAC_ABK stream");
    var cellStartX = sa.x + 11;
    var cellStartY = sa.y + 27;
    var cellGap = 1.2;
    var cellW = (sa.w - 22 - cellGap * 15) / 16;
    var cellH = 7;
    for (var row = 0; row < 4; row += 1) {
      for (var column = 0; column < 16; column += 1) {
        append(runtime, "rect", {
          x: cellStartX + column * (cellW + cellGap),
          y: cellStartY + row * (cellH + cellGap),
          width: cellW,
          height: cellH,
          rx: 1,
          "class": "sa-cell",
          "data-sa-cell": row + "-" + column,
          style: "--stream-delay:" + (row * 34 + column * 23) + "ms"
        });
      }
    }

    append(runtime, "path", { d: "M " + (sa.x + sa.w) + " " + (sa.y + sa.h / 2) + " H 1154", stroke: "#8d9caf", "stroke-width": 1.5, "marker-end": "url(#arrowhead)" });
    append(runtime, "rect", { x: 1160, y: 566, width: 115, height: 36, rx: 18, fill: kernel.split === "row" ? "#efeafe" : "#e6f5f1", stroke: kernel.split === "row" ? "#8e77d5" : "#2b9b68" });
    append(runtime, "text", { x: 1217, y: 588, fill: kernel.split === "row" ? "#5a43ad" : "#14704c", "font-size": 8.5, "font-weight": 760, "text-anchor": "middle" }, resultReduction(kernel));
    return { gb: gb, sa: sa };
  }

  function gbCapacityLabel(kernel) {
    if (kernel.kind === "sv") return "selected K-slice · ≤128 context/half";
    if (kernel.kind === "qk") return "complete q vector · 128 elements";
    return "selected K-slice · ≤256 BF16 elements";
  }

  function runtimeOperand(kernel) {
    if (kernel.kind === "qk") return "query vector";
    if (kernel.kind === "sv") return "score/context slice";
    return "activation K-slice";
  }

  function resultReduction(kernel) {
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

  function drawExecutionPhases(svg) {
    var labels = ["1 · select matrix tile", "2 · place tile in banks", "3 · load vector slice → GB", "4 · stream SA4×16 + reduce"];
    var notes = ["reset", "replay", "replay", "replay"];
    var startX = 40;
    var width = 310;
    labels.forEach(function (label, index) {
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
      append(group, "rect", { x: x, y: 742, width: width, height: 48, rx: 10, "class": "phase-pill" });
      append(group, "text", { x: x + 15, y: 763, "class": "phase-text" }, label);
      append(group, "text", { x: x + 15, y: 779, "class": "phase-note" }, notes[index] + " this stage");
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
    if (kernel.kind === "qk") {
      return [
        "K context is sequence-striped across " + number(layout.banksPerHead) + " banks per local KV head.",
        "Each bank owns " + number(layout.contextsPerBank) + " context positions at " + contextLabel(state.context) + "."
      ];
    }
    if (kernel.kind === "sv") {
      return [
        "One 128-dimension V half spans " + layout.physicalGroupChannels + " channel(s); both 8-lane halves are active.",
        number(layout.contextGroups) + " context groups/head are combined by device-local PNM."
      ];
    }
    return [
      "N is divided into 16-column BF16 bank slices; K occupies DRAM rows.",
      layout.reductionGroups > 1 ? "The same output tile is accumulated across " + layout.reductionGroups + " channel reduction groups." : "One device tile uses up to " + number(layout.totalBanks) + " banks in parallel."
    ];
  }

  function revealPhysicalMapping() {
    var token = ++state.animationToken;
    resetPhysicalVisuals();
    revealBankPanel();
    document.getElementById("physicalHint").textContent = "Phase 2/4 · tile slices are moving to channel-group banks";
    animateBankPlacement(token, function () {
      setPhase(2);
      schedule(token, reducedMotion() ? 1 : 260, function () {
        document.getElementById("runtimePanel").classList.add("is-active");
        document.getElementById("physicalHint").textContent = "Phase 3/4 · selected vector K-slice is loading into GB";
        animateVectorToGb(token, function () {
          setPhase(3);
          schedule(token, reducedMotion() ? 1 : 260, function () {
            startSaStream();
            setPhase(4);
            document.getElementById("physicalHint").textContent = "Phase 4/4 · SA4×16 is streaming · click any phase to replay";
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
    var gbSlot = document.querySelector("#physicalViz [data-gb-slot]");
    if (gbSlot) gbSlot.classList.remove("is-loading", "is-loaded");
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
    var vector = document.querySelector("#physicalViz [data-vector-packet]");
    var path = document.getElementById("vectorToGbPath");
    var slot = document.querySelector("#physicalViz [data-gb-slot]");
    var runtime = document.getElementById("runtimePanel");
    if (runtime) runtime.classList.add("is-active");
    if (path) path.classList.add("is-active");
    if (vector) setAnimatedRect(vector, 1);
    if (slot) {
      slot.classList.remove("is-loading");
      slot.classList.add("is-loaded");
    }
    setPhase(3);
  }

  function animateVectorToGb(token, callback) {
    var vector = document.querySelector("#physicalViz [data-vector-packet]");
    var path = document.getElementById("vectorToGbPath");
    var slot = document.querySelector("#physicalViz [data-gb-slot]");
    if (!vector) {
      if (callback) callback();
      return;
    }
    if (path) path.classList.add("is-active");
    if (slot) slot.classList.add("is-loading");
    var duration = reducedMotion() ? 1 : 680;
    var start = performance.now() + (reducedMotion() ? 0 : 90);
    function frame(now) {
      if (token !== state.animationToken) return;
      var t = Math.max(0, Math.min(1, (now - start) / duration));
      setAnimatedRect(vector, t);
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

  function playPhysicalStage(phase) {
    var token = ++state.animationToken;
    resetPhysicalVisuals();
    if (phase === 1) {
      document.getElementById("physicalHint").textContent = "Phase 1/4 · select the highlighted transposed-matrix tile";
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
    startSaStream();
    setPhase(4);
    document.getElementById("physicalHint").textContent = "Phase 4/4 · replaying the diagonal SA4×16 stream";
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
    if (kernel.kind === "qk") {
      facts = [
        ["KV ownership", layout.localKvHeads + " head(s)/device", "Each head retains its complete " + contextLabel(state.context) + " context."],
        ["Physical allocation", number(layout.banksPerHead) + " banks/head", layout.channelsPerHead + " channels per local KV head."],
        ["Context per bank", number(layout.contextsPerBank), "Sequence-striped K-cache positions."],
        ["QK schedule", layout.sequenceWaves + " × 2 waves", "The 128-element q vector is staged in GB, then streams through SA4×16.", "local"]
      ];
    } else if (kernel.kind === "sv") {
      facts = [
        ["SV packing", layout.svMode, "Both independent 8-lane halves are used."],
        ["128-dim group", layout.physicalGroupChannels + " channel(s)", dramLabel(state.dram) + " has " + layout.banksPerChannel + " banks/channel."],
        ["Context groups", layout.contextGroups + "/KV head", number(layout.contextsPerHalf) + " context positions per half."],
        ["SV schedule", layout.queryWaves + " query wave(s)", "GB stages ≤128 context elements/half; cross-group partials reduce locally.", "local"]
      ];
    } else {
      facts = [
        ["Device geometry", "32 × " + layout.banksPerChannel + " banks", number(layout.totalBanks) + " banks · 16 BF16 outputs/bank tile."],
        ["Channel grouping", layout.channelsPerGroup + " ch/group", layout.reductionGroups + " independent reduction group(s)."],
        ["Reduction slice", number(layout.reductionSlice), "K elements stored as DRAM-row segments per group."],
        ["Output tiling", layout.outputTiles + " tile(s)", (kernel.split === "row" ? "Produces a full [8192] partial before TP all-reduce. " : kernel.communication + " ") + "GB streams ≤256 BF16 K elements into SA4×16.", kernel.split === "row" ? "collective" : "local"]
      ];
    }
    document.getElementById("physicalFacts").innerHTML = facts.map(factHtml).join("");
  }

  function resetPhysicalMapping() {
    state.animationToken += 1;
    renderLevel3();
  }

  function exportSvg() {
    var selector = state.level === 3 ? "#physicalViz svg" : "#partitionViz svg";
    var source = document.querySelector(selector);
    if (!source) return;
    var clone = source.cloneNode(true);
    clone.setAttribute("xmlns", NS);
    clone.querySelectorAll("[tabindex]").forEach(function (node) { node.removeAttribute("tabindex"); });
    var cssText = "";
    Array.prototype.slice.call(document.styleSheets).forEach(function (sheet) {
      try {
        Array.prototype.slice.call(sheet.cssRules || []).forEach(function (rule) {
          cssText += rule.cssText + "\n";
        });
      } catch (error) {
        // Ignore stylesheets that a browser marks as cross-origin.
      }
    });
    if (cssText) {
      var embeddedStyle = svgEl("style", { type: "text/css" }, cssText);
      clone.insertBefore(embeddedStyle, clone.firstChild);
    }
    var serialized = new XMLSerializer().serializeToString(clone);
    var blob = new Blob([serialized], { type: "image/svg+xml;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "llama31-70b-" + state.kernel + "-tp" + state.tp + "-" + contextLabel(state.context).toLowerCase() + "-" + state.dram + "-level" + state.level + ".svg";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
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
          var value = key === "dram" ? raw : Number(raw);
          if (state[key] === value) return;
          state[key] = value;
          if (key === "tp" && state.device >= state.tp) state.device = 0;
          state.animationToken += 1;
          renderAll();
          if (key === "context" && state.kernel && kernels[state.kernel].kind === "dense") {
            showToast("Context changes KV-cache kernels only; this weight-matrix shape is unchanged.");
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
