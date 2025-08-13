// Best-effort jump to timestamp on the first <video> (preview)
export function jumpTo(seconds) {
  const vids = document.querySelectorAll('video');
  if (!vids?.length) return;
  const main = vids[0];
  if (main && !isNaN(seconds)) {
    main.currentTime = Number(seconds);
    main.play?.();
  }
}
