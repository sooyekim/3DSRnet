classdef Conv3D < dagnn.Filter
%%% 3D convolution layer %%%
%
% Performs 3D convolution using [1].
% Input size: [H, W, D, C, N] stored in inputs{1}
% Filter size: [kh, kw, d, C, C'] stored in params{1}
% Output size: [H, W, D, C', N] (with padding) stored in outputs{1}
% *Back-propagation (backward function) implemented*
%
% [1] https://github.com/pengsun/MexConv3D

  properties
    copy = 0 % Set copy = 1 for replicate padding 
             % (make sure no padding is applied in the d-direction in SRnet.m)
    f_sz = 3
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
  end
  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      if obj.copy
          temp = cat(3, inputs{1}(:, :, 1, :, :), inputs{1}, inputs{1}(:, :, end, :, :));
      else
          temp = inputs{1};
      end
      outputs{1} = mex_conv3d(temp, params{1}, params{2}, 'pad', obj.pad,'stride', obj.stride) ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      if obj.copy
          temp = cat(3, inputs{1}(:, :, 1, :, :), inputs{1}, inputs{1}(:, :, end, :, :));
      else
          temp = inputs{1};
      end
         [derInputs{1}, derParams{1}, derParams{2}] = mex_conv3d(...
        temp, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, 'stride', obj.stride);
    end
    
    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:3);
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(4) = obj.size(5);
    end
    
    function params = initParams(obj)
      sc = sqrt(2 / (prod(obj.size(1:4))+prod(obj.size([1 2 3 5]))));
      params{1} = randn(obj.size,'single') * sc ;
      params{2} = zeros(1,obj.size(5),'single') * sc ;
    end
    
    function obj = Conv3D(varargin)
      obj.load(varargin) ;
    end
    
  end
end