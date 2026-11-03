/* app.js - page logic for the 8-puzzle demo (vanilla JS, no libraries).
 * Depends on board.js (Puzzle, createBoardView, makeDefaultImage). */
"use strict";

(function () {
  var MAX_BYTES = 5 * 1024 * 1024;
  var ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
  var METHOD_NAMES = { qlearning: "Q-learning", dqn: "DQN", dqn2: "DQN v2", astar: "A*" };
  var REASONS = {
    step_cap: "it used up its step limit without reaching the goal",
    loop: "the policy got stuck repeating the same boards",
    unseen_state: "it reached a board it never saw during training",
    node_limit: "the search hit its node limit",
    model_unavailable: "model not trained yet",
    error: "an unexpected error occurred"
  };

  var COMPARE_ORDER = ["qlearning", "dqn", "dqn2", "astar"];

  function $(id) { return document.getElementById(id); }

  /* ---------------- app state ---------------- */
  var startBoard = null;     // the scramble the user is working on
  var mode = "watch";
  var requestToken = 0;      // bumped to ignore responses that arrive too late
  var models = {};           // /api/status result

  var mainView = createBoardView($("board"), true, onTileClick);
  var miniViews = {
    qlearning: createBoardView($("mini-qlearning"), false),
    dqn: createBoardView($("mini-dqn"), false),
    dqn2: createBoardView($("mini-dqn2"), false),
    astar: createBoardView($("mini-astar"), false)
  };

  /* ---------------- small helpers ---------------- */
  function api(path, body) {
    var opts = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    };
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) throw new Error(data.error || ("Request failed (" + r.status + ")"));
        return data;
      });
    });
  }

  function setStatus(text, kind) {
    var el = $("statusMsg");
    el.textContent = text || "";
    el.className = "msg big" + (kind ? " " + kind : "");
  }

  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function reasonText(r) { return REASONS[r] || r; }

  /* ---------------- animation (shared by Watch and Compare) ---------------- */
  var anim = { timer: null, paused: false, index: 0, total: 0, tick: null, done: null };

  function delayMs() { return 1100 - Number($("speed").value) * 100; }  // 100..1000 ms

  function setAnimButtons(running) {
    $("pauseBtn").disabled = !running;
    $("stopBtn").disabled = !running;
    $("pauseBtn").textContent = anim.paused ? "Resume" : "Pause";
  }

  function stopAnimation() {
    clearTimeout(anim.timer);
    anim.timer = null; anim.paused = false; anim.tick = null; anim.done = null;
    setAnimButtons(false);
  }

  /* Call tick(i) for i = 0..total-1, waiting delayMs() between steps, then done(). */
  function startAnimation(total, tick, done) {
    stopAnimation();
    anim.index = 0; anim.total = total; anim.tick = tick; anim.done = done;
    setAnimButtons(true);
    scheduleNext();
  }

  function scheduleNext() {
    anim.timer = setTimeout(function () {
      if (anim.paused || !anim.tick) return;
      if (anim.index >= anim.total) {
        var done = anim.done;
        stopAnimation();
        if (done) done();
        return;
      }
      anim.tick(anim.index++);
      scheduleNext();
    }, delayMs());
  }

  $("pauseBtn").addEventListener("click", function () {
    if (!anim.tick) return;
    anim.paused = !anim.paused;
    setAnimButtons(true);
    if (!anim.paused) scheduleNext();
  });

  $("stopBtn").addEventListener("click", function () {
    stopAnimation();
    requestToken++;
    showStartPosition();
    setStatus("Stopped. Board reset to the scramble.");
  });

  /* ---------------- showing boards ---------------- */
  function showStartPosition() {
    if (!startBoard) return;
    mainView.set(startBoard);
    mainView.mark(null);
    mainView.highlight(null);
    for (var m in miniViews) { miniViews[m].set(startBoard); miniViews[m].mark(null); }
  }

  /* ---------------- image handling ---------------- */
  function setImage(dataUrl) {
    document.documentElement.style.setProperty("--img", 'url("' + dataUrl + '")');
  }

  function imageMsg(text, bad) {
    var e = $("imageMsg");
    e.textContent = text;
    e.className = "msg" + (bad ? " bad" : "");
  }

  $("imageInput").addEventListener("change", function () {
    var file = this.files[0];
    if (!file) return;
    // Client-side checks (the server repeats them; never trust the browser alone).
    if (ALLOWED_TYPES.indexOf(file.type) < 0) { imageMsg("Please choose a PNG, JPEG or WEBP image.", true); this.value = ""; return; }
    if (file.size > MAX_BYTES) { imageMsg("That file is larger than 5 MB.", true); this.value = ""; return; }
    imageMsg("Uploading...");
    var form = new FormData();
    form.append("image", file);
    fetch("/api/upload", { method: "POST", body: form }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) throw new Error(data.error || ("Upload failed (" + r.status + ")"));
        setImage(data.data_url);
        imageMsg("Picture ready (" + data.width + "x" + data.height + ", cropped to a square).");
      });
    }).catch(function (err) { imageMsg(err.message, true); });
  });

  $("defaultImageBtn").addEventListener("click", function () {
    setImage(makeDefaultImage());
    $("imageInput").value = "";
    imageMsg("Using the built-in picture.");
  });

  $("numbersToggle").addEventListener("change", function () { applyNumbersToggle(); });
  function applyNumbersToggle() {
    var on = $("numbersToggle").checked;
    [$("board"), $("mini-qlearning"), $("mini-dqn"), $("mini-dqn2"), $("mini-astar")].forEach(function (b) {
      b.classList.toggle("show-nums", on);
    });
  }

  /* ---------------- scramble ---------------- */
  $("depth").addEventListener("input", function () { $("depthOut").textContent = this.value; });

  function doScramble() {
    var depth = Number($("depth").value);
    $("scrambleBtn").disabled = true;
    return api("/api/scramble", { depth: depth }).then(function (data) {
      startBoard = data.board;
      $("scrambleMsg").textContent = "Scrambled with " + depth + " random moves.";
      resetEverything();
    }).catch(function (err) {
      $("scrambleMsg").textContent = "Could not scramble: " + err.message;
    }).then(function () { $("scrambleBtn").disabled = false; });
  }
  $("scrambleBtn").addEventListener("click", doScramble);

  /* Back to a clean state for the current scramble (used on scramble and mode change). */
  function resetEverything() {
    stopAnimation();
    requestToken++;
    stopTimer();
    playState = { moves: 0, board: startBoard ? startBoard.slice() : null, solved: false, started: false };
    $("moveCount").textContent = "0";
    $("timer").textContent = "0:00";
    $("compareTable").querySelector("tbody").innerHTML = "";
    showStartPosition();
    if (mode === "play" && playState.board) mainView.set(playState.board);
    setStatus(startBoard ? "" : "Press Scramble to start.");
  }

  /* ---------------- modes ---------------- */
  var tabs = document.querySelectorAll(".tab");
  Array.prototype.forEach.call(tabs, function (tab) {
    tab.addEventListener("click", function () { setMode(tab.dataset.mode); });
  });

  function setMode(newMode) {
    mode = newMode;
    Array.prototype.forEach.call(tabs, function (t) {
      t.setAttribute("aria-pressed", t.dataset.mode === mode ? "true" : "false");
    });
    ["watch", "compare", "play"].forEach(function (m) { $("panel-" + m).hidden = m !== mode; });
    $("animControls").hidden = mode === "play";
    $("mainStage").hidden = mode === "compare";
    $("compareStage").hidden = mode !== "compare";
    resetEverything();
  }

  /* ---------------- Watch AI ---------------- */
  function updateModelButtons() {
    var notes = [];
    Array.prototype.forEach.call(document.querySelectorAll("[data-method]"), function (b) {
      var info = models[b.dataset.method];
      var ok = info && info.available;
      b.disabled = !ok;
      b.title = ok ? "" : (info ? info.message : "status unknown");
      if (info && !ok) notes.push(METHOD_NAMES[b.dataset.method] + ": " + info.message);
    });
    $("modelNote").textContent = notes.length ? "Unavailable - " + notes.join("; ") + ". A* still works." : "";
  }

  Array.prototype.forEach.call(document.querySelectorAll("[data-method]"), function (b) {
    b.addEventListener("click", function () { watch(b.dataset.method); });
  });

  function watch(method) {
    if (!startBoard) return;
    stopAnimation();
    showStartPosition();
    var token = ++requestToken;
    setStatus("Asking " + METHOD_NAMES[method] + "...");
    api("/api/solve", { board: startBoard, method: method }).then(function (res) {
      if (token !== requestToken) return;           // user did something else meanwhile
      var cur = startBoard.slice();
      setStatus(METHOD_NAMES[method] + " is playing " + res.num_moves + " move(s)...");
      startAnimation(res.moves.length, function (i) {
        cur = Puzzle.applyAction(cur, res.moves[i]);
        mainView.set(cur);
      }, function () {
        mainView.set(res.final_board);
        if (res.solved) {
          mainView.mark("solved");
          setStatus(METHOD_NAMES[method] + " solved it in " + res.num_moves + " moves (" + res.time_ms.toFixed(1) + " ms).", "ok");
        } else {
          mainView.mark("failed");
          setStatus("Failed to solve (" + reasonText(res.reason) + "). Showing the board it reached after "
            + res.num_moves + " moves.", "bad");
        }
      });
    }).catch(function (err) {
      if (token === requestToken) setStatus(err.message, "bad");
    });
  }

  /* ---------------- Compare ---------------- */
  $("compareBtn").addEventListener("click", function () {
    if (!startBoard) return;
    stopAnimation();
    showStartPosition();
    var token = ++requestToken;
    setStatus("Running all four methods...");
    api("/api/compare", { board: startBoard }).then(function (data) {
      if (token !== requestToken) return;
      var results = data.results;
      fillCompareTable(results);
      var cur = {}, longest = 0;
      COMPARE_ORDER.forEach(function (m) {
        cur[m] = startBoard.slice();
        longest = Math.max(longest, results[m].moves.length);
      });
      setStatus("Playing all four in sync...");
      startAnimation(longest, function (i) {
        // Each board advances one move per tick (if it still has moves left).
        for (var m in results) {
          if (i < results[m].moves.length) {
            cur[m] = Puzzle.applyAction(cur[m], results[m].moves[i]);
            miniViews[m].set(cur[m]);
          }
        }
      }, function () {
        for (var m in results) {
          miniViews[m].set(results[m].final_board);
          miniViews[m].mark(results[m].solved ? "solved" : "failed");
        }
        setStatus("Done. Green frame = solved, red frame = failed.");
      });
    }).catch(function (err) {
      if (token === requestToken) setStatus(err.message, "bad");
    });
  });

  function fillCompareTable(results) {
    var body = $("compareTable").querySelector("tbody");
    body.innerHTML = "";
    COMPARE_ORDER.forEach(function (m) {
      var r = results[m], tr = el("tr");
      tr.appendChild(el("td", METHOD_NAMES[m]));
      tr.appendChild(el("td", r.solved ? "Yes" : "No", r.solved ? "ok" : "bad"));
      tr.appendChild(el("td", r.solved ? String(r.num_moves) : "-"));
      tr.appendChild(el("td", r.reason === "model_unavailable" ? "-" : r.time_ms.toFixed(2)));
      var note = r.solved ? (m === "astar" ? "optimal" : "") : "Failed: " + reasonText(r.reason);
      tr.appendChild(el("td", note));
      body.appendChild(tr);
    });
  }

  /* ---------------- Play ---------------- */
  var playState = { moves: 0, board: null, solved: false, started: false };
  var timerId = null, timerStart = 0;

  function formatTime(ms) {
    var s = Math.floor(ms / 1000);
    return Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2);
  }
  function startTimer() {
    timerStart = Date.now();
    timerId = setInterval(function () { $("timer").textContent = formatTime(Date.now() - timerStart); }, 250);
  }
  function stopTimer() { clearInterval(timerId); timerId = null; }

  function onTileClick(tile) {
    if (mode !== "play") return;
    tryMove(Puzzle.slideTile(playState.board || [], tile));
  }

  /* Apply a new board from a user move (click or arrow key). */
  function tryMove(next) {
    if (!next || playState.solved) return;
    if (!playState.started) { playState.started = true; startTimer(); }
    playState.board = next;
    playState.moves++;
    $("moveCount").textContent = String(playState.moves);
    mainView.highlight(null);
    mainView.set(next);
    if (Puzzle.isGoal(next)) finishPlay();
  }

  function finishPlay() {
    playState.solved = true;
    stopTimer();
    $("timer").textContent = formatTime(Date.now() - timerStart);
    mainView.mark("solved");
    var mine = playState.moves;
    setStatus("Solved in " + mine + " moves! Checking against A*...", "ok");
    var token = requestToken;
    api("/api/solve", { board: startBoard, method: "astar" }).then(function (res) {
      if (token !== requestToken) return;
      var verdict = mine === res.num_moves ? "That is optimal!" :
        ("A* needed only " + res.num_moves + " (you used " + (mine - res.num_moves) + " extra).");
      setStatus("Solved in " + mine + " moves. Optimal for this scramble: " + res.num_moves + ". " + verdict, "ok");
    }).catch(function () {
      if (token === requestToken) setStatus("Solved in " + mine + " moves!", "ok");
    });
  }

  $("resetBtn").addEventListener("click", function () { if (startBoard) resetEverything(); });

  $("hintBtn").addEventListener("click", function () {
    if (!playState.board || playState.solved) return;
    var token = requestToken;
    api("/api/hint", { board: playState.board }).then(function (res) {
      if (token !== requestToken || res.tile === null) return;
      mainView.highlight(res.tile);
      setStatus("Hint: move tile " + res.tile + ". Optimal moves remaining: " + res.remaining + ".");
    }).catch(function (err) { setStatus(err.message, "bad"); });
  });

  // Arrow keys slide the TILE in that direction into the gap.
  document.addEventListener("keydown", function (e) {
    if (mode !== "play" || !(e.key in Puzzle.KEY_TO_ACTION)) return;
    var tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;  // sliders use arrows too
    e.preventDefault();
    if (!playState.board) return;
    tryMove(Puzzle.isLegal(playState.board, Puzzle.KEY_TO_ACTION[e.key])
      ? Puzzle.applyAction(playState.board, Puzzle.KEY_TO_ACTION[e.key]) : null);
  });

  /* ---------------- training results panel ---------------- */
  function loadResults() {
    var box = $("resultsBox");
    api("/api/results").then(function (data) {
      box.innerHTML = "";
      var hint = "Not available yet - run evaluate/visualize first.";
      if (data.rows && data.rows.length) {
        var wrap = el("div", undefined, "table-wrap"), table = el("table");
        var cols = ["method", "solve_rate", "avg_moves", "avg_optimal", "optimality_gap", "avg_time_ms"];
        var head = el("tr");
        cols.forEach(function (c) { head.appendChild(el("th", c)); });
        table.appendChild(el("thead")).appendChild(head);
        var tb = el("tbody");
        data.rows.forEach(function (row) {
          var tr = el("tr");
          cols.forEach(function (c) {
            var v = row[c];
            tr.appendChild(el("td", typeof v === "number" ? String(Math.round(v * 1000) / 1000) : (v === null || v === undefined ? "-" : String(v))));
          });
          tb.appendChild(tr);
        });
        table.appendChild(tb); wrap.appendChild(table); box.appendChild(wrap);
      } else {
        box.appendChild(el("p", "Comparison table: " + hint, "note"));
      }
      var labels = { "solve_by_depth.png": "Solve rate by scramble depth", "learning_curves.png": "Learning curves",
        "snapshots_by_depth.png": "Solve rate by training snapshot and depth" };
      Object.keys(labels).forEach(function (name) {
        if (data.images[name]) {
          var img = el("img", undefined, "results-img");
          img.src = "/api/results/image/" + name;
          img.alt = labels[name];
          box.appendChild(img);
        } else {
          box.appendChild(el("p", labels[name] + ": " + hint, "note"));
        }
      });
    }).catch(function () {
      box.textContent = "Could not load results. Run evaluate/visualize first.";
    });
  }

  /* ---------------- startup ---------------- */
  setImage(makeDefaultImage());
  applyNumbersToggle();
  mainView.set([1, 2, 3, 4, 5, 6, 7, 8, 0]);
  Object.keys(miniViews).forEach(function (m) { miniViews[m].set([1, 2, 3, 4, 5, 6, 7, 8, 0]); });
  api("/api/status").then(function (s) { models = s; updateModelButtons(); })
    .catch(function () { $("modelNote").textContent = "Could not read model status."; });
  loadResults();
  doScramble();
})();
