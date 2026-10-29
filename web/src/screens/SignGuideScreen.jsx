import { useEffect, useMemo, useState } from "react";
import { getVocab } from "../api.js";

const VIDEO_COUNTS = {
  alright: 3,
  "good-morning": 3,
  "good-afternoon": 3,
  "good-evening": 3,
  "good-night": 3,
  hello: 4,
  "how-are-you": 3,
  pleased: 3,
  "thank-you": 3,
};

function signSlug(label) {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export default function SignGuideScreen({ onPractice }) {
  const [signs, setSigns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [activeVideo, setActiveVideo] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getVocab()
      .then((items) => {
        if (!cancelled) setSigns(items);
      })
      .catch((requestError) => {
        console.error("Could not load sign guide vocabulary:", requestError);
        if (!cancelled) setError("Could not load the sign list. Please try again.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredSigns = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return signs;
    return signs.filter((sign) =>
      `${sign.label} ${sign.category}`.toLowerCase().includes(normalizedQuery)
    );
  }, [query, signs]);

  return (
    <section className="mx-auto max-w-6xl">
      <div className="mb-3">
        <p className="mb-1 text-xs font-bold uppercase tracking-[0.18em] text-[#8b6a38]">
          Supported signs
        </p>
        <h2 className="text-2xl font-bold tracking-tight text-[#1d2c25] sm:text-3xl">
          Sign guide
        </h2>
        <p className="mt-1 max-w-2xl text-sm leading-snug text-[#69776e]">
          Look up a supported message and choose an example to watch its sign.
        </p>
      </div>

      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <label className="block min-w-0 flex-1">
          <span className="mb-1 block text-xs font-medium text-[#52635a]">
            Find a message
          </span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search the supported signs"
            className="w-full rounded-lg border border-[#d8ddd7] bg-white px-3 py-2 text-sm text-[#26362e] placeholder:text-[#9aa39c]"
          />
        </label>
        {!loading && !error && (
          <p className="pb-2 text-xs font-semibold text-[#7b877e] sm:whitespace-nowrap">
            {filteredSigns.length} of {signs.length} messages
          </p>
        )}
      </div>

      {loading ? (
        <p className="rounded-xl border border-[#e6e3db] bg-white p-5 text-sm text-[#69776e]" role="status">
          Loading supported signs…
        </p>
      ) : error ? (
        <p className="rounded-xl border border-[#f1c6bc] bg-[#fff1ed] p-4 text-sm text-[#9b3c2f]" role="alert">
          {error}
        </p>
      ) : filteredSigns.length ? (
        <>
          <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {filteredSigns.map((sign) => (
              <li
                key={sign.label}
                className="flex flex-col rounded-xl border border-[#e6e3db] bg-white p-3 shadow-[0_4px_16px_rgba(36,49,40,0.04)] sm:p-4"
              >
                <span className="mb-2 w-fit rounded-full bg-[#edf5ee] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.1em] text-[#356449]">
                  {sign.category}
                </span>
                <h3 className="text-lg font-bold tracking-tight text-[#26362e]">
                  {sign.label}
                </h3>
                <div className="mt-2 border-t border-[#eeede7] pt-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="mr-1 text-[11px] font-medium text-[#748078]">Examples</span>
                    {Array.from(
                      { length: VIDEO_COUNTS[signSlug(sign.label)] || 0 },
                      (_, index) => {
                        const exampleNumber = index + 1;
                        const selected =
                          activeVideo?.slug === signSlug(sign.label) &&
                          activeVideo.number === exampleNumber;
                        return (
                          <button
                            key={exampleNumber}
                            type="button"
                            aria-pressed={selected}
                            onClick={() =>
                              setActiveVideo(
                                selected
                                  ? null
                                  : {
                                      slug: signSlug(sign.label),
                                      label: sign.label,
                                      number: exampleNumber,
                                    }
                              )
                            }
                            className={`rounded-md border px-2 py-1 text-[11px] font-semibold transition focus-visible:ring ${
                              selected
                                ? "border-[#1c5b46] bg-[#edf5ee] text-[#1c5b46]"
                                : "border-[#dddcd4] bg-[#fbfaf7] text-[#52635a] hover:border-[#a9c4ad]"
                            }`}
                          >
                            Example {exampleNumber}
                          </button>
                        );
                      }
                    )}
                  </div>
                  {activeVideo?.slug === signSlug(sign.label) && (
                    <div className="mt-3 overflow-hidden rounded-xl bg-[#17211c]">
                      <video
                        key={`${activeVideo.slug}-${activeVideo.number}`}
                        controls
                        playsInline
                        preload="none"
                        className="block aspect-video w-full"
                        aria-label={`${activeVideo.label} gesture example ${activeVideo.number}`}
                      >
                        <source
                          src={`/sign_guide/${activeVideo.slug}/${String(
                            activeVideo.number
                          ).padStart(2, "0")}.mp4`}
                          type="video/mp4"
                        />
                        Your browser does not support MP4 video playback.
                      </video>
                      <button
                        type="button"
                        onClick={() => setActiveVideo(null)}
                        className="m-2 rounded-md px-2 py-1 text-xs font-medium text-white/80 hover:bg-white/10 hover:text-white focus-visible:ring"
                      >
                        Close example
                      </button>
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="rounded-xl border border-dashed border-[#d8d9d1] bg-white p-6 text-center text-sm text-[#748078]">
          No supported messages match “{query}”.
        </p>
      )}

      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-xl bg-[#1c5b46] p-3 text-white sm:p-4">
        <div>
          <h3 className="font-semibold">Ready to try one?</h3>
          <p className="mt-0.5 text-xs text-white/75 sm:text-sm">
            Return to the camera and hold Space or the capture button while you sign.
          </p>
        </div>
        <button
          type="button"
          onClick={onPractice}
          className="rounded-lg bg-white px-4 py-2.5 text-sm font-semibold text-[#1c5b46] transition hover:bg-[#edf5ee] focus-visible:ring"
        >
          Go to Capture
        </button>
      </div>
    </section>
  );
}
