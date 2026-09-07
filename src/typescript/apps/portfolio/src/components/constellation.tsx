const stars = [
  [86, 72, 5],
  [182, 38, 3],
  [244, 128, 4],
  [338, 76, 3],
  [414, 164, 5],
  [374, 278, 3],
  [274, 344, 5],
  [158, 308, 4],
  [70, 220, 3],
  [216, 220, 3],
  [318, 202, 3],
  [120, 156, 2],
] as const;

const lines = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [8, 0],
  [0, 9],
  [9, 2],
  [9, 10],
  [10, 4],
  [10, 6],
  [9, 7],
  [11, 0],
  [11, 9],
] as const;

export function Constellation({ compact = false }: { compact?: boolean }) {
  return (
    <svg
      className={compact ? "constellation constellation-compact" : "constellation"}
      viewBox="0 0 480 390"
      aria-hidden="true"
      focusable="false"
    >
      <g className="constellation-lines">
        {lines.map(([from, to]) => {
          const a = stars[from];
          const b = stars[to];
          return <line key={`${from}-${to}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />;
        })}
      </g>
      <g className="constellation-stars">
        {stars.map(([x, y, radius], index) => (
          <circle
            key={`${x}-${y}`}
            cx={x}
            cy={y}
            r={radius}
            style={{ animationDelay: `${index * 110}ms` }}
          />
        ))}
      </g>
    </svg>
  );
}
