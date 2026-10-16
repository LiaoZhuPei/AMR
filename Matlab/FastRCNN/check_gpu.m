if gpuDeviceCount > 0
    dev = gpuDevice();
    fprintf('正在使用GPU: %s, 显存: %.1f GB\n', dev.Name, dev.TotalMemory/1e9);
end