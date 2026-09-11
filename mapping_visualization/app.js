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

  function renderMatrixPartition(kernel) {
    var host = document.getElementById("partitionViz");
    host.setAttribute("aria-label", kernel.title + " global matrix partitioned across " + state.tp + " devices");
    var svg = svgRoot(host, "0 0 1200 560");
    addArrowDefs(svg);
    var shape = globalShape(kernel);
    var local = localShape(kernel);
    var frame = fitRect(shape.k, shape.n, { x: 55, y: 118, w: 410, h: 270, minW: 145, minH: 92 });
    var positions = devicePositions(state.tp);

    append(svg, "text", { x: 55, y: 48, "class": "svg-kicker" }, "GLOBAL LOGICAL MATRIX");
    append(svg, "text", { x: 55, y: 72, "class": "svg-title" }, kernel.globalLabel);
    append(svg, "text", { x: 55, y: 89, "class": "svg-subtitle" },
      kernel.split === "column" ? "Column parallel · split output N" : "Row parallel · split reduction K");

    append(svg, "rect", {
      x: frame.x,
      y: frame.y,
      width: frame.w,
      height: frame.h,
      rx: 7,
      "class": "matrix-frame"
    });

    append(svg, "text", {
      x: frame.x + frame.w / 2,
      y: frame.y + frame.h + 27,
      "class": "svg-small",
      "text-anchor": "middle"
    }, kernel.outputAxis);
    append(svg, "text", {
      x: frame.x - 23,
      y: frame.y + frame.h / 2,
      "class": "svg-small",
      "text-anchor": "middle",
      transform: "rotate(-90 " + (frame.x - 23) + " " + (frame.y + frame.h / 2) + ")"
    }, kernel.inputAxis);

    for (var i = 0; i < state.tp; i += 1) {
      var color = DEVICE_COLORS[i];
      var source = sourcePiece(frame, i, kernel.split === "column" ? "column" : "row", state.tp);
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
      var splitText = kernel.split === "column" ? "Output N / columns" : kernel.split === "row" ? "Reduction K / rows" : "KV-head groups";
      facts = [
        ["Global operand", "[" + number(shape.k) + ", " + number(shape.n) + "]", kernel.globalLabel],
        ["TP partition axis", splitText, kernel.split === "column" ? "Vertical matrix slices" : "Horizontal matrix slices"],
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
      result.representativeChannels = [0];
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
    document.getElementById("level3-subtitle").textContent = "Inspect how the TP-local operand is tiled over " + dramLabel(state.dram) + " channels, banks and rows.";
    document.getElementById("physicalEquation").textContent = "Device " + state.device + " local operand " + localShapeLabel(kernel);
    document.getElementById("physicalHint").textContent = "Click the highlighted tile";
    renderPhysicalSvg(kernel, layout);
    renderPhysicalFacts(kernel, layout);
  }

  function renderPhysicalSvg(kernel, layout) {
    var host = document.getElementById("physicalViz");
    host.setAttribute("aria-label", kernel.title + " physical mapping inside Device " + state.device);
    var svg = svgRoot(host, "0 0 1280 720");
    addArrowDefs(svg);
    var color = DEVICE_COLORS[state.device];
    var local = localShape(kernel);
    var frame = fitRect(local.k, local.n, { x: 65, y: 165, w: 390, h: 285, minW: 155, minH: 100 });

    append(svg, "text", { x: 54, y: 49, "class": "svg-kicker" }, "TP-LOCAL OPERAND");
    append(svg, "text", { x: 54, y: 75, "class": "svg-title" }, "Device " + state.device + " · " + localShapeLabel(kernel));
    append(svg, "text", { x: 54, y: 94, "class": "svg-subtitle" }, physicalOperandSubtitle(kernel, layout));

    append(svg, "rect", { x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 8, "class": "matrix-frame" });
    drawLocalMatrixGrid(svg, frame, kernel, layout, color);

    append(svg, "text", { x: frame.x + frame.w / 2, y: frame.y + frame.h + 27, "class": "svg-small", "text-anchor": "middle" }, kernel.outputAxis || "output dimension");
    append(svg, "text", {
      x: frame.x - 23,
      y: frame.y + frame.h / 2,
      "class": "svg-small",
      "text-anchor": "middle",
      transform: "rotate(-90 " + (frame.x - 23) + " " + (frame.y + frame.h / 2) + ")"
    }, kernel.inputAxis || "reduction dimension");

    var selected = selectedTile(frame, kernel, layout);
    var tileGroup = append(svg, "g", {
      "class": "matrix-tile-clickable",
      role: "button",
      tabindex: "0",
      "aria-label": "Map selected tile to DRAM banks"
    });
    append(tileGroup, "rect", {
      x: selected.x,
      y: selected.y,
      width: selected.w,
      height: selected.h,
      rx: 4,
      "class": "selected-tile"
    });
    textLines(tileGroup, selected.x + selected.w / 2, selected.y + selected.h / 2 - 4, selected.lines, "svg-mono", 13, "middle");
    append(tileGroup, "text", {
      x: selected.x + selected.w / 2,
      y: selected.y + selected.h / 2 + 27,
      fill: "#1744a5",
      "font-size": 8,
      "font-weight": 760,
      "text-anchor": "middle"
    }, "CLICK TO MAP");
    tileGroup.addEventListener("click", revealPhysicalMapping);
    tileGroup.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") revealPhysicalMapping();
    });

    append(svg, "path", {
      d: "M " + (selected.x + selected.w) + " " + (selected.y + selected.h / 2) +
        " C 500 " + (selected.y + selected.h / 2) + ", 520 245, 625 245",
      "class": "mapping-path",
      id: "mappingPath"
    });

    var bankPanel = append(svg, "g", { id: "bankPanel", "class": "bank-panel-svg" });
    append(bankPanel, "rect", { x: 575, y: 38, width: 655, height: 520, rx: 16, fill: "#fbfcfe", stroke: "#cdd6e3" });
    append(bankPanel, "text", { x: 600, y: 68, "class": "svg-kicker" }, "PHYSICAL DEVICE");
    append(bankPanel, "text", { x: 600, y: 91, "class": "svg-title" }, dramLabel(state.dram) + " · 32 channels × " + layout.banksPerChannel + " banks");
    append(bankPanel, "text", { x: 600, y: 108, "class": "svg-subtitle" }, bankPanelSubtitle(kernel, layout));

    var bankGeometry = drawBankGrid(bankPanel, kernel, layout, color);
    drawRuntimePanel(bankPanel, kernel, layout);
    drawPackets(svg, selected, bankGeometry, layout);
    drawExecutionPhases(svg);

    append(svg, "text", { x: 54, y: 535, "class": "svg-kicker" }, "SELECTED TILE RULE");
    textLines(svg, 54, 559, selectedRuleLines(kernel, layout), "svg-subtitle", 17);

    svg.setAttribute("data-bank-count", layout.banksPerChannel);
  }

  function physicalOperandSubtitle(kernel, layout) {
    if (kernel.kind === "qk") {
      return layout.localKvHeads + " local KV head(s) · complete " + contextLabel(state.context) + " context per head";
    }
    if (kernel.kind === "sv") {
      return layout.localKvHeads + " local KV head(s) · dual-half V-cache packing";
    }
    return kernel.split === "column" ? "TP-local output-column shard" : "TP-local reduction-row shard";
  }

  function drawLocalMatrixGrid(svg, frame, kernel, layout, color) {
    var rows;
    var columns;
    if (kernel.kind === "dense") {
      rows = Math.min(layout.reductionGroups, 32);
      columns = Math.min(8, Math.max(2, Math.ceil(layout.localN / Math.max(layout.outputCapacity, 1) * 4)));
    } else {
      rows = Math.min(layout.localKvHeads || 1, 8);
      columns = kernel.kind === "qk" ? 8 : 4;
    }
    for (var r = 1; r < rows; r += 1) {
      append(svg, "line", { x1: frame.x, y1: frame.y + frame.h * r / rows, x2: frame.x + frame.w, y2: frame.y + frame.h * r / rows, "class": "matrix-grid-line" });
    }
    for (var c = 1; c < columns; c += 1) {
      append(svg, "line", { x1: frame.x + frame.w * c / columns, y1: frame.y, x2: frame.x + frame.w * c / columns, y2: frame.y + frame.h, "class": "matrix-grid-line" });
    }
    append(svg, "rect", { x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 8, fill: "none", stroke: color.strong, "stroke-width": 1.5 });
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
        x: frame.x + frame.w * 0.08,
        y: frame.y + frame.h * 0.06,
        w: frame.w * 0.84,
        h: Math.max(60, frame.h * 0.24),
        lines: ["one context group", number(layout.contextsPerHalf) + " × 128"]
      };
    }
    var tileFraction = Math.min(1, layout.outputCapacity / Math.max(layout.localN, 1));
    var groupFraction = 1 / Math.max(layout.reductionGroups, 1);
    return {
      x: frame.x,
      y: frame.y,
      w: Math.max(78, frame.w * tileFraction),
      h: Math.max(62, frame.h * groupFraction),
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
    var x = 655;
    var y = 144;
    var width = 535;
    var rowHeight = 11.4;
    var gap = 2;
    var cellW = (width - gap * (layout.banksPerChannel - 1)) / layout.banksPerChannel;
    var targets = [];

    append(parent, "text", { x: 600, y: y - 13, "class": "svg-small" }, "CHANNEL");
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
      append(parent, "text", { x: 627, y: cy + 7.5, "class": "svg-small", "text-anchor": "end" }, "C" + channel);
      append(parent, "rect", { x: 635, y: cy, width: 8, height: 8.6, rx: 2, fill: groupColor.strong, opacity: activeBank(kernel, layout, channel) ? 0.82 : 0.16 });
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
    return { targets: targets, x: x, y: y, width: width, cellW: cellW, rowHeight: rowHeight };
  }

  function drawRuntimePanel(parent, kernel, layout) {
    var runtime = append(parent, "g", { id: "runtimePanel", "class": "runtime-panel" });
    append(runtime, "line", { x1: 690, y1: 527, x2: 1160, y2: 527, stroke: "#d5dde8" });
    append(runtime, "text", { x: 600, y: 548, "class": "svg-kicker" }, "RUNTIME DATAFLOW");
    append(runtime, "rect", { x: 690, y: 474, width: 155, height: 48, rx: 9, "class": "gb-box" });
    append(runtime, "text", { x: 768, y: 495, fill: "#7d4b14", "font-size": 11, "font-weight": 760, "text-anchor": "middle" }, "GB · runtime operand");
    append(runtime, "text", { x: 768, y: 510, fill: "#a46928", "font-size": 8, "text-anchor": "middle" }, runtimeOperand(kernel));
    append(runtime, "path", { d: "M 845 498 H 925", stroke: "#8d9caf", "stroke-width": 1.4, "marker-end": "url(#arrowhead)" });
    append(runtime, "rect", { x: 930, y: 474, width: 105, height: 48, rx: 9, "class": "sa-box" });
    append(runtime, "text", { x: 982, y: 496, fill: "#08675f", "font-size": 12, "font-weight": 780, "text-anchor": "middle" }, "SA 4×16");
    append(runtime, "text", { x: 982, y: 511, fill: "#328f86", "font-size": 8, "text-anchor": "middle" }, "MAC_ABK");
    append(runtime, "path", { d: "M 1035 498 H 1101", stroke: "#8d9caf", "stroke-width": 1.4, "marker-end": "url(#arrowhead)" });
    append(runtime, "rect", { x: 1107, y: 480, width: 95, height: 36, rx: 18, fill: kernel.split === "row" ? "#efeafe" : "#e6f5f1", stroke: kernel.split === "row" ? "#8e77d5" : "#2b9b68" });
    append(runtime, "text", { x: 1154, y: 502, fill: kernel.split === "row" ? "#5a43ad" : "#14704c", "font-size": 8.5, "font-weight": 760, "text-anchor": "middle" }, resultReduction(kernel));
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

  function drawPackets(svg, selected, bankGeometry, layout) {
    var targets = bankGeometry.targets;
    if (!targets.length) return;
    targets.forEach(function (target, index) {
      var sourceW = Math.max(3, selected.w / targets.length);
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
      node.style.fill = DEVICE_COLORS[(state.device + target.channel) % DEVICE_COLORS.length].fill;
    });
  }

  function drawExecutionPhases(svg) {
    var labels = ["1 · select matrix tile", "2 · place bank slices", "3 · stream operand to GB", "4 · SA compute & reduce"];
    var startX = 54;
    var width = 276;
    labels.forEach(function (label, index) {
      var x = startX + index * (width + 13);
      append(svg, "rect", { x: x, y: 661, width: width, height: 32, rx: 8, "class": "phase-pill", "data-phase": index + 1 });
      append(svg, "text", { x: x + width / 2, y: 681, "class": "phase-text", "text-anchor": "middle", "data-phase-text": index + 1 }, label);
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
    var panel = document.getElementById("bankPanel");
    if (!panel || panel.classList.contains("is-revealed")) return;
    state.animationToken += 1;
    var token = state.animationToken;
    panel.classList.add("is-revealed");
    document.getElementById("mappingPath").classList.add("is-active");
    document.getElementById("physicalHint").textContent = "Tile slices are moving to bank rows";
    setPhase(1);

    var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var packets = Array.prototype.slice.call(document.querySelectorAll("#physicalViz [data-packet]"));
    var duration = reduced ? 1 : 780;
    var stagger = reduced ? 0 : 22;
    var start = performance.now() + (reduced ? 0 : 240);

    document.querySelectorAll("#physicalViz [data-active-bank='true']").forEach(function (cell, index) {
      cell.style.animationDelay = (reduced ? 0 : 350 + (index % 32) * 6 + Math.floor(index / 32) * 7) + "ms";
      cell.classList.add("is-active");
    });

    function packetFrame(now) {
      if (token !== state.animationToken) return;
      var complete = true;
      packets.forEach(function (node, index) {
        var t = Math.max(0, Math.min(1, (now - start - index * stagger) / duration));
        if (t < 1) complete = false;
        var eased = 1 - Math.pow(1 - t, 3);
        node.style.opacity = t > 0 ? "1" : "0";
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
        requestAnimationFrame(packetFrame);
      } else {
        setPhase(2);
        schedule(token, reduced ? 1 : 350, function () {
          document.getElementById("runtimePanel").classList.add("is-active");
          setPhase(3);
        });
        schedule(token, reduced ? 2 : 760, function () {
          setPhase(4);
          document.getElementById("physicalHint").textContent = "Mapping complete · replay or choose another configuration";
        });
      }
    }
    requestAnimationFrame(packetFrame);
  }

  function schedule(token, delay, callback) {
    window.setTimeout(function () {
      if (token === state.animationToken) callback();
    }, delay);
  }

  function setPhase(phase) {
    document.querySelectorAll("#physicalViz [data-phase], #physicalViz [data-phase-text]").forEach(function (node) {
      var value = Number(node.getAttribute("data-phase") || node.getAttribute("data-phase-text"));
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
        ["QK schedule", layout.sequenceWaves + " × 2 waves", "Sequence waves × two SA4 query waves.", "local"]
      ];
    } else if (kernel.kind === "sv") {
      facts = [
        ["SV packing", layout.svMode, "Both independent 8-lane halves are used."],
        ["128-dim group", layout.physicalGroupChannels + " channel(s)", dramLabel(state.dram) + " has " + layout.banksPerChannel + " banks/channel."],
        ["Context groups", layout.contextGroups + "/KV head", number(layout.contextsPerHalf) + " context positions per half."],
        ["SV schedule", layout.queryWaves + " query wave(s)", "Cross-group partials are reduced locally.", "local"]
      ];
    } else {
      facts = [
        ["Device geometry", "32 × " + layout.banksPerChannel + " banks", number(layout.totalBanks) + " banks · 16 BF16 outputs/bank tile."],
        ["Channel grouping", layout.channelsPerGroup + " ch/group", layout.reductionGroups + " independent reduction group(s)."],
        ["Reduction slice", number(layout.reductionSlice), "K elements stored as DRAM-row segments per group."],
        ["Output tiling", layout.outputTiles + " tile(s)", kernel.split === "row" ? "Produces a full [8192] partial before TP all-reduce." : kernel.communication, kernel.split === "row" ? "collective" : "local"]
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
