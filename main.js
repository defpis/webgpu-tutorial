(async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return;
  const device = await adapter.requestDevice();

  const module = device.createShaderModule({
    label: "doubling compute module",
    code: `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;

      @compute @workgroup_size(1) fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
        data[id.x] = data[id.x] * 2.0;
      }
    `,
  });

  const pipeline = device.createComputePipeline({
    label: "doubling compute pipeline",
    layout: "auto",
    compute: { module, entryPoint: "compute" },
  });

  const input = new Float32Array([1, 3, 5]);

  const workBuffer = device.createBuffer({
    label: "work buffer",
    size: input.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(workBuffer, 0, input);

  const bindGroup = device.createBindGroup({
    label: "bindGroup for work buffer",
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: workBuffer } }],
  });

  const resultBuffer = device.createBuffer({
    label: "result buffer",
    size: input.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder({ label: "doubling encoder" });
  const pass = encoder.beginComputePass({ label: "doubling compute pass" });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(input.length);
  pass.end();

  encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size);

  const commands = encoder.finish();
  device.queue.submit([commands]);

  await resultBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(resultBuffer.getMappedRange().slice());
  resultBuffer.unmap();

  console.log("input", [...input]);
  console.log("result", [...result]);
})();
