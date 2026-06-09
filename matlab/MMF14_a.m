classdef MMF14_a < PROBLEM
% <2018> <multi> <real> <multimodal>
% Multi-modal multi-objective test function
%------------------------------- Reference --------------------------------
% Y. Liu, G. G. Yen, and D. Gong, A Multi-Modal Multi-Objective 
% Evolutionary Algorithm Using Two-Archive and Recombination Strategies, 
% IEEE Transactions on Evolutionary Computation, 2018, 23(4): 660-674.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        POS;    % Pareto optimal set for IGDX calculation
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            if isempty(obj.M); obj.M = 3; end
            if isempty(obj.D); obj.D = 3; end
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,X)
            [N,D] = size(X);
            M = obj.M;
            num_of_peak = 2;
            
            % 建立变量间的非线性关联
            % x_g = x3 - 0.5*sin(pi*x2)
            x_g = X(:,end) - 0.5*sin(pi*X(:,end-1));
            
            % g 函数计算 (包含平移项 1/(2*num_of_peak))
            g = 2 - (sin(num_of_peak * pi * (x_g + 1/(2*num_of_peak)))).^2;
            
            % 球面坐标变换 (Spherical transformation)
            PopObj = repmat(1+g,1,M) .* fliplr(cumprod([ones(N,1), cos(X(:,1:M-1)*pi/2)], 2)) .* [ones(N,1), sin(X(:,M-1:-1:1)*pi/2)];
        end
    end
end