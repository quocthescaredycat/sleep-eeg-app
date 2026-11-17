import type { Library, LibraryEpoch, Similar, SimilaritySpace, SpaceName } from '../api/client'
import { useLibrary, useLibraryEpochs, useSimilar } from '../api/queries'
import { WaveformChart } from '../charts/WaveformChart'
import { STAGE_COLORS, STAGE_NAMES, formatClock, formatPercent } from '../lib/stages'
import { ErrorMessage } from './ErrorMessage'
import type { ModelResult } from './Results'

interface Props {
  recordingId: string
  results: ModelResult[]
  selected: number
  /** Space named in the URL, if any. */
  requested: string | null
  /** Model shown in detail: its space is the default. */
  activeModelId: string
  onSpace: (space: SpaceName) => void
}

/** Why a space cannot be searched for this recording, or null when it can. */
function blocked(space: SimilaritySpace, results: ModelResult[]): string | null {
  if (!space.available) return space.reason ?? 'Not available on this server.'
  if (space.model_id && !results.some((result) => result.model.id === space.model_id)) {
    return `Run ${space.label} on this recording to search its space.`
  }
  return null
}

function StageName({ stage }: { stage: number }) {
  return (
    <span className="inline-flex items-center gap-1 font-medium">
      <span
        className="inline-block size-2 rounded-full"
        style={{ backgroundColor: STAGE_COLORS[stage] }}
        aria-hidden
      />
      {STAGE_NAMES[stage]}
    </span>
  )
}

function formatSimilarity(similar: Similar, score: number): string {
  return similar.score === 'cosine_similarity' ? `cos ${score.toFixed(3)}` : `dist ${score.toFixed(2)}`
}

function Neighbour({
  similar,
  rank,
  waveform,
  fold,
}: {
  similar: Similar
  rank: number
  waveform: LibraryEpoch | undefined
  fold: number | null
}) {
  const { epoch, score } = similar.neighbours[rank - 1]
  return (
    <li
      className="rounded-lg border border-slate-200 p-3 text-xs"
      data-testid="neighbour"
      data-participant={epoch.participant}
      data-stage={STAGE_NAMES[epoch.stage]}
    >
      <div className="flex items-baseline justify-between gap-2 whitespace-nowrap">
        <span className="font-medium text-slate-800">
          #{rank} · {epoch.recording}
        </span>
        <span
          className="tabular-nums text-slate-500"
          title={
            similar.score === 'cosine_similarity'
              ? 'Cosine similarity: 1 is identical'
              : 'Euclidean distance between z-scored band powers: 0 is identical'
          }
        >
          {formatSimilarity(similar, score)}
        </span>
      </div>
      {waveform ? (
        <WaveformChart
          waveform={waveform}
          label={`Filtered EEG of library epoch ${epoch.recording} ${formatClock(epoch.onset_seconds)}`}
          compact
          testId="neighbour-waveform"
        />
      ) : (
        <div className="h-24" aria-hidden />
      )}
      <p className="mt-1 text-sm">
        Expert <StageName stage={epoch.stage} />
      </p>
      <p className="text-slate-600" title="Predicted by a fold that never trained on this participant">
        Transformer {STAGE_NAMES[epoch.transformer_stage]} · AttnSleep{' '}
        {STAGE_NAMES[epoch.attnsleep_stage]}
      </p>
      <p className="mt-1 text-slate-500">
        {formatClock(epoch.onset_seconds)} · {epoch.participant} · {epoch.split}
        {fold != null &&
          (epoch.in_embedding_training ? ` · fold ${fold} trained on it` : ` · unseen by fold ${fold}`)}
      </p>
    </li>
  )
}

function SpaceSummary({ similar, space }: { similar: Similar; space: SimilaritySpace }) {
  const vote = similar.vote
  if (!vote) return <p className="text-sm text-slate-600">No neighbours found.</p>
  const precision = space.evaluation?.vote_precision[vote.stage]
  return (
    <p className="text-sm" data-testid="similar-summary">
      <strong>
        {vote.count} of {similar.neighbours.length}
      </strong>{' '}
      neighbours were scored <StageName stage={vote.stage} /> by experts.
      {precision != null && space.evaluation && (
        <>
          {' '}
          In the library evaluation, a {space.evaluation.k}-NN vote for {STAGE_NAMES[vote.stage]} in
          this space matched the expert {formatPercent(precision, 0)} of the time.
        </>
      )}
    </p>
  )
}

function BiasNote({ space, spaces }: { space: SimilaritySpace; spaces: SimilaritySpace[] }) {
  const disagreement = space.evaluation?.disagreement
  if (!disagreement) return null
  const sides = Object.entries(disagreement.sides_with)
    .map(([name, share]) => {
      const label = spaces.find((candidate) => candidate.name === name)?.label ?? name
      return `${label} ${formatPercent(share, 0)}`
    })
    .join(', ')
  return (
    <p className="text-xs text-slate-600" data-testid="similar-bias">
      Bias check: on the {disagreement.n_epochs.toLocaleString()} evaluation epochs where the
      models disagree, this space&apos;s vote sided with {sides} of the time and matched the expert{' '}
      {formatPercent(disagreement.matches_expert, 0)}.
      {space.model_id
        ? ' A model space tends to confirm its own model: compare with the signal space.'
        : ' The signal space is shaped by no model.'}
    </p>
  )
}

function Votes({
  library,
  spaces,
  answers,
}: {
  library: Library
  spaces: SimilaritySpace[]
  answers: (Similar | undefined)[]
}) {
  const votes = spaces.flatMap((space, index) => {
    const vote = answers[index]?.vote
    return vote ? [{ space, vote, k: answers[index]?.neighbours.length ?? 0 }] : []
  })
  if (votes.length < 2) return null
  const unanimous = votes.every((entry) => entry.vote.stage === votes[0].vote.stage)
  const everySpace = votes.length === library.spaces.length
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2 text-sm" data-testid="similar-votes">
      <span className="label mr-2">Vote per space</span>
      {votes.map(({ space, vote, k }, index) => (
        <span key={space.name}>
          {index > 0 && ' · '}
          {space.label}: <StageName stage={vote.stage} /> ({vote.count}/{k})
        </span>
      ))}
      <p className="mt-1 text-xs text-slate-600">
        {unanimous
          ? `All ${votes.length} spaces agree.` +
            (everySpace && library.unanimous
              ? ` In the library evaluation, a unanimous vote (${formatPercent(library.unanimous.coverage, 0)} of epochs) matched the expert ${formatPercent(library.unanimous.accuracy, 0)} of the time.`
              : '')
          : 'The spaces disagree: treat this epoch as ambiguous.'}
      </p>
    </div>
  )
}

/**
 * Expert-scored library epochs nearest to the selected epoch, in a space the researcher
 * picks: either model's features, or band powers that no model has shaped.
 */
export function SimilarPanel(props: Props) {
  const { recordingId, results, selected, requested, activeModelId, onSpace } = props
  const library = useLibrary()
  const spaces = library.data?.spaces ?? []
  const searchable = spaces.filter((space) => blocked(space, results) === null)
  const answers = useSimilar(
    recordingId,
    selected,
    searchable.map((space) => space.name),
  )
  const active =
    searchable.find((space) => space.name === requested) ??
    searchable.find((space) => space.model_id === activeModelId) ??
    searchable[0]
  const answer = active ? answers[searchable.indexOf(active)] : undefined
  const waveforms = useLibraryEpochs(answer?.data?.neighbours.map((item) => item.epoch.id) ?? [])

  if (library.isPending) return null
  const status = library.data
  if (!status?.available) {
    return (
      <section className="card" aria-labelledby="similar-title" data-testid="similar-panel">
        <h2 id="similar-title" className="card-title">
          Similar epochs
        </h2>
        <p className="mt-1 text-sm text-slate-600" data-code={status?.code ?? 'REQUEST_FAILED'}>
          Similarity search is not available on this server.{' '}
          {status?.message ?? 'The library status could not be loaded.'}
        </p>
      </section>
    )
  }

  const stages = results.map((result) => result.predictions.stages[selected])
  const disagree = results.length > 1 && stages[0] !== stages[1]
  const similar = answer?.data

  return (
    <section className="card space-y-3" aria-labelledby="similar-title" data-testid="similar-panel">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="similar-title" className="card-title mr-auto">
          Similar epochs in the reference library
        </h2>
        <span className="label">Find similar using</span>
        <div role="radiogroup" aria-label="Find similar using" className="flex flex-wrap gap-1">
          {spaces.map((space) => {
            const reason = blocked(space, results)
            const checked = space === active
            return (
              <button
                key={space.name}
                type="button"
                role="radio"
                aria-checked={checked}
                disabled={reason !== null}
                title={reason ?? undefined}
                className={`button ${checked ? 'button-primary' : ''}`}
                onClick={() => onSpace(space.name)}
              >
                {space.label}
              </button>
            )
          })}
        </div>
      </div>
      <p className="text-xs text-slate-500">
        {status.n_epochs?.toLocaleString()} expert-scored Sleep-EDF epochs from{' '}
        {status.n_participants} participants ({status.library_id}, vectors of fold {status.fold}).
        Neighbours are evidence to inspect, not a diagnosis; the expert stage is the reference for
        library epochs only.
        {status.synthetic && (
          <span className="ml-1 rounded bg-amber-100 px-1.5 py-0.5 font-medium text-amber-800">
            synthetic test library
          </span>
        )}
      </p>
      {similar?.library_match && (
        <p className="text-sm text-slate-700" data-testid="library-match">
          This recording is library night {similar.library_match.recording}: neighbours from its
          participant {similar.library_match.participant} are left out.
        </p>
      )}
      {disagree && (
        <p
          className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
          data-testid="similar-disagreement"
        >
          {results[0].model.name} says {STAGE_NAMES[stages[0]]} and {results[1].model.name} says{' '}
          {STAGE_NAMES[stages[1]]} for this epoch. Compare the neighbours in each space.
        </p>
      )}
      <Votes
        library={status}
        spaces={searchable}
        answers={answers.map((query) => query.data)}
      />
      {!active && (
        <p className="text-sm text-slate-600">
          {spaces.map((space) => blocked(space, results)).find(Boolean)}
        </p>
      )}
      {active && answer?.error ? <ErrorMessage error={answer.error} /> : null}
      {active && !similar && !answer?.error && (
        <p className="text-sm text-slate-500">Searching the library…</p>
      )}
      {active && similar && (
        <>
          <SpaceSummary similar={similar} space={active} />
          <BiasNote space={active} spaces={spaces} />
          <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5" aria-label="Nearest library epochs">
            {similar.neighbours.map((item, index) => (
              <Neighbour
                key={item.epoch.id}
                similar={similar}
                rank={item.rank}
                waveform={waveforms[index]?.data}
                fold={status.fold}
              />
            ))}
          </ol>
        </>
      )}
    </section>
  )
}

