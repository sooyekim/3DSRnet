classdef Bicubic_PS < dagnn.ElementWise
%%% Bicubic resizing + inverse pixel shuffle (space-to-depth) layer %%%
%
% Performs bicubic resizing on the middle frame of inputs{1}, inverse pixel 
% shuffle (space-to-depth operation) on the result, and stores it in outputs{1}.
% Input size: [H, W, 5, 1, N]
% Output size: [H, W, 1, scale*scale, N]
% *Back-propagation (backward function) not implemented*
    properties
        scale = 2;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            sz = size(inputs{1}); % [H, W, 5, 1, N]
            if size(sz, 2) < 4
                sz(4:5) = [1, 1];
            end
            Bic_ch = gpuArray(zeros(sz(1), sz(2), 1, obj.scale*obj.scale, sz(5))); 
           
            Bic_im =imresize(inputs{1}(:, :, 3, 1, :), obj.scale,'bicubic'); % [H*scale, W*scale, 1, 1, N]
            % inverse pixel shuffle / space-to-depth ([H, W, 1, scale*scale, N])
            for c = 1:obj.scale*obj.scale
                q = floor((c-1)/obj.scale)+1;
                r = mod(c, obj.scale);
                if r == 0, r = obj.scale; end
                Bic_ch(:, :, 1, c, :) = Bic_im(q:obj.scale:end, r:obj.scale:end, 1, 1, :);
            end
            outputs{1}=Bic_ch;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = 0*inputs{1} ;
            derParams = {} ;
        end
        
        function obj = my_bicubic(varargin)
            obj.load(varargin) ;
            obj.scale = obj.scale;
        end
    end
end
