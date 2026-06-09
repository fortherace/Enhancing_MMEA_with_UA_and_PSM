classdef Omnitest < PROBLEM
    % <multi> <real> <multimodal>
    % Omni-test problem

    properties
        ideal_point
        nadir_point
    end

    methods
        %% 默认设置
        function Setting(obj)
            obj.M        = 2;          % 目标数
            obj.D        = 2;          % 决策变量维度（可改）
            obj.lower    = zeros(1,obj.D);   % 下界 0
            obj.upper    = 6*ones(1,obj.D);  % 上界 6
            obj.encoding = ones(1,obj.D);    % 实数编码
            
            % 理想点与 Nadir 点
            obj.ideal_point = [-2,-2];
            obj.nadir_point = [0,0];
        end
        
        %% 计算目标函数
        function PopObj = CalObj(obj,X)

            F1 = sum(sin(pi * X), 2);
            F2 = sum(cos(pi * X), 2);
            
            PopObj = [F1, F2];
        end
    end
end