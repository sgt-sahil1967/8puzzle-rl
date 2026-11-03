/* board.js - puzzle helpers and board rendering (no dependencies).
 *
 * A board is an array of 9 ints, row by row, 0 = blank.
 * Actions move the BLANK: 0 up, 1 down, 2 left, 3 right (same as puzzle/env.py).
 */
"use strict";

var Puzzle = (function () {
  var DELTAS = [-3, 3, -1, 1];          // index offset of the blank per action
  var GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0];

  /* True if moving the blank by `action` stays on the 3x3 grid. */
  function isLegal(board, action) {
    var b = board.indexOf(0);
    if (action === 0) return b >= 3;
    if (action === 1) return b <= 5;
    if (action === 2) return b % 3 !== 0;
    return b % 3 !== 2;
  }

  /* Return a new board after moving the blank (the neighbour tile slides in). */
  function applyAction(board, action) {
    if (!isLegal(board, action)) return board.slice();
    var next = board.slice();
    var b = board.indexOf(0);
    var t = b + DELTAS[action];
    next[b] = next[t];
    next[t] = 0;
    return next;
  }

  function isGoal(board) {
    return board.every(function (v, i) { return v === GOAL[i]; });
  }

  /* Is the tile at index `idx` directly next to the blank? */
  function isAdjacentToBlank(board, idx) {
    var b = board.indexOf(0);
    var dr = Math.abs(Math.floor(b / 3) - Math.floor(idx / 3));
    var dc = Math.abs((b % 3) - (idx % 3));
    return dr + dc === 1;
  }

  /* Slide `tile` into the blank if adjacent; returns new board or null. */
  function slideTile(board, tile) {
    var idx = board.indexOf(tile);
    if (tile === 0 || idx < 0 || !isAdjacentToBlank(board, idx)) return null;
    var next = board.slice();
    next[board.indexOf(0)] = tile;
    next[idx] = 0;
    return next;
  }

  /* Arrow key -> action. The arrow moves the TILE, so the blank moves the opposite way. */
  var KEY_TO_ACTION = { ArrowLeft: 3, ArrowRight: 2, ArrowUp: 1, ArrowDown: 0 };

  return { applyAction: applyAction, isGoal: isGoal, slideTile: slideTile,
           isLegal: isLegal, KEY_TO_ACTION: KEY_TO_ACTION, DELTAS: DELTAS };
})();

/* Create (or rebuild) a board view inside `host`.
 *  interactive=true -> tiles are real <button>s that call onTile(tileNumber).
 *  interactive=false -> plain decorative <div>s (mini boards in Compare mode).
 * Tile t (1..8) shows image cell row=(t-1)//3, col=(t-1)%3 via background-position.
 * The picture itself comes from the CSS variable --img set on <html>. */
function createBoardView(host, interactive, onTile) {
  host.innerHTML = "";
  var tiles = {};
  for (var t = 1; t <= 8; t++) {
    var el = document.createElement(interactive ? "button" : "div");
    if (interactive) el.type = "button";
    else el.setAttribute("aria-hidden", "true");
    el.className = "tile";
    el.dataset.num = String(t);
    var row = Math.floor((t - 1) / 3), col = (t - 1) % 3;
    el.style.backgroundPosition = (col * 50) + "% " + (row * 50) + "%";
    if (interactive) {
      el.setAttribute("aria-label", "Tile " + t);
      (function (tile) {
        el.addEventListener("click", function () { if (onTile) onTile(tile); });
      })(t);
    }
    host.appendChild(el);
    tiles[t] = el;
  }

  return {
    host: host,
    tiles: tiles,
    /* Move every tile to its cell for this board. */
    set: function (board) {
      for (var i = 0; i < 9; i++) {
        var tile = board[i];
        if (tile === 0) continue;
        var el = tiles[tile];
        el.style.left = ((i % 3) * 100 / 3) + "%";
        el.style.top = (Math.floor(i / 3) * 100 / 3) + "%";
      }
    },
    /* Highlight one tile (or clear with null). */
    highlight: function (tile) {
      for (var k in tiles) tiles[k].classList.remove("hint");
      if (tile) tiles[tile].classList.add("hint");
    },
    /* Mark the frame green/red/none. */
    mark: function (state) {
      host.classList.toggle("solved", state === "solved");
      host.classList.toggle("failed", state === "failed");
    }
  };
}

/* Build a colourful default picture on a canvas (used before any upload). */
function makeDefaultImage() {
  var size = 600, c = document.createElement("canvas");
  c.width = c.height = size;
  var g = c.getContext("2d");
  var grad = g.createLinearGradient(0, 0, size, size);
  grad.addColorStop(0, "#ff7a59"); grad.addColorStop(0.5, "#ffd166"); grad.addColorStop(1, "#4ecdc4");
  g.fillStyle = grad; g.fillRect(0, 0, size, size);
  g.globalAlpha = 0.55;
  var colors = ["#3b6df0", "#ef476f", "#06d6a0", "#ffffff"];
  for (var i = 0; i < 14; i++) {
    g.fillStyle = colors[i % colors.length];
    g.beginPath();
    g.arc((i * 137) % size, (i * 211) % size, 40 + (i * 23) % 80, 0, Math.PI * 2);
    g.fill();
  }
  g.globalAlpha = 1;
  g.fillStyle = "rgba(20,20,40,.85)";
  g.font = "bold 150px system-ui, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText("8", size / 2, size / 2);
  return c.toDataURL("image/png");
}
