/**
 * AudioWorklet Processor for PCM extraction
 *
 * Receives microphone audio, resamples to target rate if needed,
 * accumulates samples into chunks, and sends them to the main thread.
 *
 * Messages sent to main thread:
 *   - { type: 'pcm-chunk', buffer: ArrayBuffer }  — Int16 PCM data (every chunkDurationMs)
 *   - { type: 'volume', volume: number }           — RMS volume level 0-1 (every frame)
 *   - { type: 'diagnostic', ... }                  — Debug info (first 10 frames + periodic)
 *
 * Reference: GoogleChromeLabs/web-audio-samples (worklet-recorder)
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()

    // Configuration from processorOptions
    const opts = options.processorOptions || {}
    this.targetSampleRate = opts.targetSampleRate || 16000
    this.chunkDurationMs = opts.chunkDurationMs || 300
    this.inputSampleRate = opts.inputSampleRate || sampleRate // AudioContext.sampleRate

    // Calculate samples per chunk at target rate
    this.samplesPerChunk = Math.floor(
      (this.targetSampleRate * this.chunkDurationMs) / 1000
    )

    // Accumulation buffer (Float32 at target sample rate)
    this.buffer = new Float32Array(this.samplesPerChunk * 2) // 2x for safety
    this.bufferOffset = 0

    // Resampling state
    this.needsResample = this.inputSampleRate !== this.targetSampleRate
    this.resampleRatio = this.targetSampleRate / this.inputSampleRate

    // Running flag
    this.running = true

    // Diagnostic counters
    this._frameCount = 0
    this._nonZeroFrames = 0

    // Listen for stop command
    this.port.onmessage = (event) => {
      if (event.data.type === 'stop') {
        this.running = false
      }
    }
  }

  process(inputs, outputs, parameters) {
    if (!this.running) return false

    const input = inputs[0]
    // Guard against empty inputs — Firefox may deliver empty arrays
    // for the first 10-70 frames before the MediaStream starts producing.
    // Reference: https://bugzilla.mozilla.org/show_bug.cgi?id=1629478
    if (!input || input.length === 0) return true

    // Take first channel (mono)
    const channelData = input[0]
    if (!channelData || channelData.length === 0) return true

    // Calculate volume (RMS) and peak
    let sumSquares = 0
    let maxAbs = 0
    for (let i = 0; i < channelData.length; i++) {
      sumSquares += channelData[i] * channelData[i]
      const abs = Math.abs(channelData[i])
      if (abs > maxAbs) maxAbs = abs
    }
    const rms = Math.sqrt(sumSquares / channelData.length)
    this.port.postMessage({ type: 'volume', volume: rms })

    // Diagnostic: log first 10 frames and then every 500th frame
    this._frameCount++
    if (maxAbs > 0.001) this._nonZeroFrames++
    if (this._frameCount <= 10 || this._frameCount % 500 === 0) {
      this.port.postMessage({
        type: 'diagnostic',
        frame: this._frameCount,
        inputLen: channelData.length,
        rms: rms.toFixed(6),
        maxAbs: maxAbs.toFixed(6),
        nonZeroFrames: this._nonZeroFrames,
        totalFrames: this._frameCount,
        sampleSnippet: Array.from(channelData.slice(0, 5)).map(v => v.toFixed(6)),
      })
    }

    // Resample or copy directly
    let samples
    if (this.needsResample) {
      samples = this._resample(channelData)
    } else {
      samples = channelData
    }

    // Accumulate into buffer
    for (let i = 0; i < samples.length; i++) {
      if (this.bufferOffset >= this.buffer.length) {
        // Buffer overflow protection - flush what we have
        this._flushChunk()
      }
      this.buffer[this.bufferOffset++] = samples[i]
    }

    // Check if we have a complete chunk
    while (this.bufferOffset >= this.samplesPerChunk) {
      this._flushChunk()
    }

    return true
  }

  _flushChunk() {
    const samplesToSend = Math.min(this.bufferOffset, this.samplesPerChunk)
    if (samplesToSend === 0) return

    // Convert Float32 (-1.0 to 1.0) to Int16 (-32768 to 32767)
    const int16 = new Int16Array(samplesToSend)
    for (let i = 0; i < samplesToSend; i++) {
      const s = Math.max(-1, Math.min(1, this.buffer[i]))
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }

    // Send PCM chunk as transferable ArrayBuffer
    this.port.postMessage(
      { type: 'pcm-chunk', buffer: int16.buffer },
      [int16.buffer]
    )

    // Shift remaining samples to beginning of buffer
    const remaining = this.bufferOffset - samplesToSend
    if (remaining > 0) {
      this.buffer.copyWithin(0, samplesToSend, this.bufferOffset)
    }
    this.bufferOffset = remaining
  }

  _resample(inputSamples) {
    // Simple linear interpolation resampling
    const outputLength = Math.floor(inputSamples.length * this.resampleRatio)
    const output = new Float32Array(outputLength)

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i / this.resampleRatio
      const srcIndexFloor = Math.floor(srcIndex)
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputSamples.length - 1)
      const frac = srcIndex - srcIndexFloor

      output[i] =
        inputSamples[srcIndexFloor] * (1 - frac) +
        inputSamples[srcIndexCeil] * frac
    }

    return output
  }
}

registerProcessor('pcm-processor', PCMProcessor)
