function multipleMatrixOnCPU(m, k, n, a, b) {
  const r = Array.from({ length: m * n }, () => 0);

  for (let mi = 0; mi < m; mi++) {
    for (let ni = 0; ni < n; ni++) {
      for (let ki = 0; ki < k; ki++) {
        r[mi * n + ni] += a[mi * k + ki] * b[ki * n + ni];
      }
    }
  }

  return r;
}

function random() {
  return Math.floor(Math.random() * 100);
}

function randomMatrix(m, n) {
  return Array.from({ length: m * n }, () => random());
}

async function time(fn) {
  const start = performance.now();
  const result = await fn();
  console.log(performance.now() - start);
  return result;
}

async function multipleMatrixOnGPU(m, k, n, a, b) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return;
  const device = await adapter.requestDevice();

  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<uniform> s: vec3<f32>;
      @group(0) @binding(1) var<storage, read> a: array<f32>;
      @group(0) @binding(2) var<storage, read> b: array<f32>;
      @group(0) @binding(3) var<storage, read_write> r: array<f32>;

      @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let m = u32(s.x);
        let k = u32(s.y);
        let n = u32(s.z);

        let mi = id.x;
        let ni = id.y;

        if (mi >= m || ni >= n) {
          return;
        }

        let ri = mi * n + ni;

        for (var ki = 0u; ki < k; ki = ki + 1u) {
          r[ri] = r[ri] + a[mi * k + ki] * b[ki * n + ni];
        }
      }
    `,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const sGPUBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: 3 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM,
  });
  const sBuffer = sGPUBuffer.getMappedRange();
  new Float32Array(sBuffer).set([m, k, n]);
  sGPUBuffer.unmap();

  const aGPUBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: m * k * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const aBuffer = aGPUBuffer.getMappedRange();
  new Float32Array(aBuffer).set(a);
  aGPUBuffer.unmap();

  const bGPUBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: k * n * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const bBuffer = bGPUBuffer.getMappedRange();
  new Float32Array(bBuffer).set(b);
  bGPUBuffer.unmap();

  const resultSize = m * n * Float32Array.BYTES_PER_ELEMENT;

  const rGPUBuffer = device.createBuffer({
    size: resultSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: sGPUBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: aGPUBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: bGPUBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: rGPUBuffer,
        },
      },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(m / 8), Math.ceil(n / 8));
  pass.end();

  const resultGPUBuffer = device.createBuffer({
    size: resultSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  encoder.copyBufferToBuffer(rGPUBuffer, 0, resultGPUBuffer, 0, resultSize);

  const commands = encoder.finish();
  device.queue.submit([commands]);

  await resultGPUBuffer.mapAsync(GPUMapMode.READ);
  const result = Array.from(
    new Float32Array(resultGPUBuffer.getMappedRange().slice())
  );
  resultGPUBuffer.unmap();

  return result;
}

(async function main() {
  const m = 64;
  const k = 32;
  const n = 64;

  const a = randomMatrix(m, k);
  const b = randomMatrix(k, n);

  const r1 = await time(async () => multipleMatrixOnCPU(m, k, n, a, b));
  const r2 = await time(async () => await multipleMatrixOnGPU(m, k, n, a, b));

  const equal = r1.every((n, i) => n === r2[i]);
  console.log("r1 == r2", equal);
})();
