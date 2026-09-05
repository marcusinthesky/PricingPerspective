import { Reveal } from "@/components/reveal";
import { sitePath } from "@/lib/site";

const appearances = [
  {
    name: "World Finance Conference",
    href: "https://www.world-finance-conference.com/",
    className: "appearance-world-finance",
    content: (
      <img src={sitePath("/logos/world-finance-conference.png")} alt="World Finance Conference" />
    ),
  },
  {
    name: "arXiv",
    href: "https://arxiv.org/",
    className: "appearance-arxiv",
    content: "arXiv",
  },
  {
    name: "SSRN",
    href: "https://www.ssrn.com/",
    className: "appearance-ssrn",
    content: "SSRN",
  },
  {
    name: "Hugging Face",
    href: "https://huggingface.co/",
    className: "appearance-hugging-face",
    content: "Hugging Face",
  },
  {
    name: "University of Cape Town",
    href: "https://www.uct.ac.za/",
    className: "appearance-uct",
    content: (
      <img src={sitePath("/logos/uct-horizontal-black.svg")} alt="University of Cape Town" />
    ),
  },
] as const;

export function AppearingIn() {
  return (
    <section className="appearing-in site-shell" aria-labelledby="appearing-in-title">
      <h2 className="appearing-in-title" id="appearing-in-title">
        As appearing in
      </h2>
      <Reveal className="appearance-list">
        <div className="appearance-track animate-[appearance-marquee_32s_linear_infinite] motion-reduce:animate-none">
          {[false, true].map((isDuplicate) => (
            <div
              className="appearance-group"
              aria-hidden={isDuplicate || undefined}
              key={String(isDuplicate)}
            >
              {appearances.map((appearance) => (
                <a
                  className={`appearance-link ${appearance.className}`}
                  href={appearance.href}
                  key={appearance.name}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={appearance.name}
                  tabIndex={isDuplicate ? -1 : undefined}
                >
                  {appearance.content}
                </a>
              ))}
            </div>
          ))}
        </div>
      </Reveal>
    </section>
  );
}
