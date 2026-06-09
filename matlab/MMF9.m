classdef MMF9 < PROBLEM
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
            obj.M = 2;
            obj.D = 2;
            obj.lower    = [0.1, 0.1];
            obj.upper    = [1.1, 1.1];
            obj.encoding = ones(1,obj.D);
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,X)
            PopObj(:,1) = X(:,1);
            g = 2 - sin(2*pi*X(:,2)).^6;
            PopObj(:,2) = g./X(:,1);
        end
    end
end