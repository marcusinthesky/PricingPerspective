/* oxlint-disable jsx-a11y/prefer-tag-over-role -- Inline SVG needs an image role to group its accessible title and description. */
import { fieldContours } from "@/components/field-geometry";

const footprints = [
  { id: "a", x: 230, y: 238, angle: -32, rx: 56, ry: 29, label: "Firm A · μ₁", lx: 95, ly: 192 },
  { id: "b", x: 328, y: 313, angle: 38, rx: 78, ry: 31, label: "Firm B · μ₂", lx: 380, ly: 260 },
  { id: "c", x: 427, y: 395, angle: -24, rx: 45, ry: 27, label: "Firm C · μ₃", lx: 443, ly: 465 },
];

/** Server-rendered artwork. Only CSS animates; the checkbox is a native pause control. */
export function EconomicField() {
  return (
    <figure className="economic-field" aria-labelledby="field-caption">
      <div className="field-heading">
        <span>01 / Economic field</span>
        <span>Schematic</span>
      </div>
      <label className="field-motion">
        <input type="checkbox" aria-controls="economic-field-plate" />
        Pause motion
      </label>
      <svg
        id="economic-field-plate"
        className="field-plate"
        viewBox="0 0 640 620"
        role="img"
        aria-labelledby="field-title field-description"
      >
        <title id="field-title">A shared field, different exposures</title>
        <desc id="field-description">
          Three extended firm footprints span different regions of a continuous economic field. A
          shared oscillation carries the field, firm footprints and their labels together. A small
          radar dial and a waveform phase marker show the cycle. Contours and motion are schematic,
          not estimated data or a simulation of the research model.
        </desc>
        <defs>
          <pattern id="field-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0H0V40" className="field-grid-line" />
          </pattern>
          <pattern id="field-samples" width="9" height="9" patternUnits="userSpaceOnUse">
            <circle cx="3" cy="3" r=".85" className="field-sample" />
          </pattern>
          <radialGradient id="field-wash">
            <stop className="field-wash-center" />
            <stop offset="1" className="field-wash-edge" />
          </radialGradient>
          <clipPath id="field-window">
            <rect x="26" y="30" width="588" height="552" />
          </clipPath>
        </defs>
        <rect x="26" y="30" width="588" height="552" fill="url(#field-grid)" />
        <path
          className="field-neatline"
          d="M26 54V30H50M590 30H614V54M26 558V582H50M590 582H614V558"
        />
        <g clipPath="url(#field-window)">
          <g className="field-drift">
            <ellipse
              className="field-wash field-cycle"
              cx="233"
              cy="240"
              rx="205"
              ry="221"
              fill="url(#field-wash)"
            />
            <ellipse
              className="field-wash field-cycle"
              cx="414"
              cy="375"
              rx="180"
              ry="150"
              fill="url(#field-wash)"
            />
            <g className="field-contours">
              {fieldContours.map((d, index) => (
                <path key={d} d={d} className={index % 3 === 0 ? "field-isobath" : undefined} />
              ))}
            </g>
            {footprints.map((firm) => (
              <g key={firm.id}>
                <g transform={`translate(${firm.x} ${firm.y}) rotate(${firm.angle})`}>
                  <ellipse className="field-footprint" rx={firm.rx} ry={firm.ry} />
                  <ellipse
                    className="field-exposure field-cycle"
                    rx={firm.rx - 6}
                    ry={firm.ry - 6}
                  />
                  <ellipse rx={firm.rx - 7} ry={firm.ry - 7} fill="url(#field-samples)" />
                  <path className="field-registration" d="M-5 0H5M0-5V5" />
                </g>
                <path
                  className="field-label-leader"
                  d={
                    firm.id === "a"
                      ? "M164 199H184L201 216"
                      : firm.id === "b"
                        ? "M415 267H388L369 287"
                        : "M458 447V436L445 421"
                  }
                />
                <text className="field-measure" x={firm.lx} y={firm.ly}>
                  {firm.label}
                </text>
              </g>
            ))}
          </g>
          <g className="field-radar" transform="translate(537 105)">
            <circle r="43" />
            <circle r="27" />
            <circle r="11" />
            <path d="M-49 0H49M0-49V49" />
            <g className="field-radar-sweep">
              <path className="field-radar-sector" d="M0 0L0-43A43 43 0 0 1 30.4-30.4Z" />
              <path className="field-radar-hand" d="M0 0V-43" />
            </g>
            <circle className="field-radar-origin" r="2" />
          </g>
        </g>
        <g className="field-axis">
          <text x="28" y="607">
            INFORMATION SPACE
          </text>
          <path d="M494 603H613M494 599V607M534 599V607M574 599V607M613 599V607" />
        </g>
      </svg>
      <div className="field-oscillation" aria-hidden="true">
        <span>Field oscillation</span>
        <svg viewBox="0 0 240 44" className="field-waveform">
          <title>Illustrative field oscillation</title>
          <path className="field-trace-baseline" d="M20 22H220" />
          <path
            className="field-trace"
            d="M40 22C56.8 22 63.2 6 80 6S103.2 22 120 22S143.2 38 160 38S183.2 22 200 22"
          />
          <g className="field-phase-progress">
            <path className="field-phase-guide" d="M40 2V42" />
            <g className="field-phase-value">
              <circle className="field-phase-dot" cx="40" cy="22" r="3" />
            </g>
          </g>
        </svg>
        <span>Illustrative</span>
      </div>
      <figcaption id="field-caption">
        <strong>A shared field. Different exposures.</strong>
        <span>Each firm spans a distribution of economic activities.</span>
      </figcaption>
      <div className="field-legend" aria-hidden="true">
        <span>
          <i className="field-legend-contour" /> Shared field
        </span>
        <span>
          <i className="field-legend-footprint" /> Firm footprint
        </span>
      </div>
    </figure>
  );
}
