import { useMemo } from 'react'

import { useEChart } from './useEChart'

const COLORS = ['#0f766e', '#7c3aed']

/** Any 30 s window: an uploaded recording's epoch or a library epoch. */
export interface Waveform {
  sampling_rate: number
  channels: { label: string; samples: number[] }[]
}

interface Props {
  waveform: Waveform
  /** Accessible name of the chart. */
  label: string
  /** Small, axis-free rendering for thumbnails. */
  compact?: boolean
  testId?: string
}

/** Symmetric y-range in steps of 50 µV, so neighbouring epochs are easy to compare. */
function amplitudeRange(samples: number[]): number {
  let peak = 0
  for (const value of samples) peak = Math.max(peak, Math.abs(value))
  return Math.max(50, Math.ceil(peak / 50) * 50)
}

export function WaveformChart({ waveform, label, compact = false, testId = 'waveform-chart' }: Props) {
  const option = useMemo(() => {
    const tops = compact ? ['2%', '51%'] : ['8%', '54%']
    const height = compact ? '47%' : '36%'
    return {
      animation: false,
      tooltip: compact
        ? { show: false }
        : { trigger: 'axis', valueFormatter: (value: number) => `${value.toFixed(1)} µV` },
      grid: tops.map((top) =>
        compact ? { left: 4, right: 4, top, height } : { left: 56, right: 16, top, height },
      ),
      xAxis: waveform.channels.map((channel, index) => ({
        gridIndex: index,
        type: 'value',
        min: 0,
        max: channel.samples.length / waveform.sampling_rate,
        interval: 5,
        show: !compact,
        axisLabel: { show: index === waveform.channels.length - 1, formatter: '{value} s' },
      })),
      yAxis: waveform.channels.map((channel, index) => {
        const range = amplitudeRange(channel.samples)
        return {
          gridIndex: index,
          type: 'value',
          min: -range,
          max: range,
          interval: range,
          show: !compact,
          name: `${channel.label.replace('EEG ', '')} (µV)`,
          nameLocation: 'middle',
          nameGap: 40,
        }
      }),
      series: waveform.channels.map((channel, index) => ({
        name: channel.label,
        type: 'line',
        xAxisIndex: index,
        yAxisIndex: index,
        showSymbol: false,
        silent: compact,
        sampling: 'lttb',
        lineStyle: { width: compact ? 0.75 : 1, color: COLORS[index % COLORS.length] },
        itemStyle: { color: COLORS[index % COLORS.length] },
        data: channel.samples.map((value, sample) => [sample / waveform.sampling_rate, value]),
      })),
    }
  }, [waveform, compact])
  const container = useEChart(option)
  return (
    <div
      ref={container}
      className={compact ? 'h-24 w-full' : 'h-72 w-full'}
      role="img"
      aria-label={label}
      data-testid={testId}
    />
  )
}
