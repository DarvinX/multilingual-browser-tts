function half(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31;
  const fraction = bits & 1023;
  if (!exponent) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

export function expandWeights(packed) {
  const view = new DataView(packed);
  const headerBytes = view.getUint32(0, true);
  const { entries, rawBytes, groupSize } = JSON.parse(
    new TextDecoder().decode(new Uint8Array(packed, 4, headerBytes)),
  );
  const weights = new Uint8Array(rawBytes);
  const out = new DataView(weights.buffer);
  let input = 4 + headerBytes;
  for (const [format, offset, bytes, size] of entries) {
    if (format === 'raw') {
      if (bytes !== size) throw new Error('The model download is incomplete.');
      weights.set(new Uint8Array(packed, input, size), offset);
    } else {
      const count = bytes / 4;
      if (format === 'fp16') {
        for (let i = 0; i < count; i++) out.setFloat32(offset + 4 * i, half(view.getUint16(input + 2 * i, true)), true);
      } else {
        const codesPerByte = format === 'int4' ? 2 : 1;
        const groups = Math.ceil(count / groupSize);
        let scales = input + groups * groupSize / codesPerByte;
        for (let base = 0; base < count; base += groupSize) {
          const scale = half(view.getUint16(scales, true));
          scales += 2;
          for (let i = base; i < Math.min(base + groupSize, count); i++) {
            const code = format === 'int4' ? ((view.getUint8(input + (i >>> 1)) >>> (i % 2 * 4)) & 15) - 8
              : view.getInt8(input + i);
            out.setFloat32(offset + 4 * i, code * scale, true);
          }
        }
      }
    }
    input += size;
  }
  if (input !== packed.byteLength) throw new Error('The model download is incomplete.');
  return weights;
}
