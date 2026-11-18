import { expandWeights as expandOriginal } from './weights.js';

export function expandWeights(packed) {
  const headerBytes = new DataView(packed).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(packed, 4, headerBytes)));
  if (!header.shuffled) return expandOriginal(packed);

  const source = new Uint8Array(packed);
  const restored = source.slice();
  let offset = 4 + headerBytes;
  for (const [format, , decodedBytes, length] of header.entries) {
    let codes = 0;
    if (format === 'int8' || format === 'int4') {
      const groups = Math.ceil(decodedBytes / 4 / header.groupSize);
      codes = groups * header.groupSize / (format === 'int4' ? 2 : 1);
    }
    const width = format === 'raw' ? 4 : 2;
    const count = (length - codes) / width;
    const start = offset + codes;
    for (let byte = 0; byte < width; byte++) {
      for (let i = 0; i < count; i++) {
        restored[start + width * i + byte] = source[start + count * byte + i];
      }
    }
    offset += length;
  }
  if (offset !== source.length) throw new Error('The model download is incomplete.');
  return expandOriginal(restored.buffer);
}
