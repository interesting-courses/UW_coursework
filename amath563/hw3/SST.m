clear all; close all; clc;

sst = ncread('sst.wkmean.1990-present.nc','sst');
mask = ncread('lsmask.nc','mask');

N = 360*180;
L = 1400;

for j = 1:L
    sst_mask(:,:,j) = sst(:,:,j).*mask;
    imagesc(sst_mask(:,:,j)')
    colormap jet
    axis equal off
    pause(0.02)
end

