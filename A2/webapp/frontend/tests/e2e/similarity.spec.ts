import { ATTNSLEEP, TRANSFORMER, expect, test } from './fixtures'

const STAGE = '(W|N1|N2|N3|REM)'

/**
 * Needs a searchable library: CI builds a synthetic one from the bundled sample
 * (backend/scripts/build_library.py synthetic); a local stack can serve the real one.
 * Uploads are the bundled sample, which is library night SC4031 in both.
 */
test.describe('similar epochs', () => {
  test.beforeEach(async ({ api }) => {
    const library = await api.library()
    test.skip(!library.available, `similarity search is unavailable (${library.code})`)
  })

  test('neighbours come from other participants in the chosen space', async ({
    api,
    recordingPage,
    page,
  }) => {
    const recording = await api.createRecording({
      label: 'similar',
      hypnogram: true,
      models: [TRANSFORMER, ATTNSLEEP],
    })
    await api.waitForJobs(recording.id)
    await recordingPage.goto(recording.id, '?epoch=10')

    // The model shown in detail sets the default space.
    await expect(recordingPage.spaceOption('Transformer')).toHaveAttribute('aria-checked', 'true')
    await expect(recordingPage.neighbours).toHaveCount(5)
    await expect(recordingPage.neighbourWaveforms).toHaveCount(5)
    await expect(recordingPage.similarSummary).toContainText(
      new RegExp(`\\d of 5 neighbours were scored ${STAGE} by experts`),
    )
    // Re-uploading a library night must not return the epoch itself.
    await expect(recordingPage.libraryMatch).toContainText('library night SC4031')
    expect(await recordingPage.neighbourParticipants()).not.toContain('SC403')
    for (const space of ['Transformer', 'AttnSleep', 'Signal (model-free)']) {
      await expect(recordingPage.similarVotes).toContainText(`${space}:`)
    }

    await recordingPage.spaceOption('Signal (model-free)').click()
    await expect(page).toHaveURL(/[?&]space=signal/)
    await expect(recordingPage.spaceOption('Signal (model-free)')).toHaveAttribute(
      'aria-checked',
      'true',
    )
    await expect(recordingPage.neighbours).toHaveCount(5)
    expect(await recordingPage.neighbourParticipants()).not.toContain('SC403')

    await page.reload()
    await expect(recordingPage.spaceOption('Signal (model-free)')).toHaveAttribute(
      'aria-checked',
      'true',
    )
    await recordingPage.next.click()
    await expect(recordingPage.epochSummary).toContainText('Epoch 11')
    await expect(recordingPage.neighbours).toHaveCount(5)
  })

  test('a model space needs that model to have run', async ({ api, recordingPage }) => {
    const recording = await api.createRecording({ label: 'similar-one-model', models: [TRANSFORMER] })
    await api.waitForJobs(recording.id)
    await recordingPage.goto(recording.id)

    const attnsleep = recordingPage.spaceOption('AttnSleep')
    await expect(attnsleep).toBeDisabled()
    await expect(attnsleep).toHaveAttribute('title', /Run AttnSleep on this recording/)
    await expect(recordingPage.spaceOption('Signal (model-free)')).toBeEnabled()
    await expect(recordingPage.neighbours).toHaveCount(5)
  })
})
