classdef SYMPART < PROBLEM
    % <multi> <real> <multimodal>
    % SYMPART multimodal multi-objective test problem

    properties
        a = 1;
        b = 10;
        c = 10;
        w = 0;          % rotation angle
        IRM             % inverted rotation matrix
        ideal_point
        nadir_point
    end

    methods
        %% 默认设置
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 2;
            
            r = max(obj.b,obj.c);
            obj.lower    = -10*r*ones(1,obj.D);
            obj.upper    =  10*r*ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
            
            % 计算逆旋转矩阵
            obj.IRM = [cos(obj.w) sin(obj.w); ...
                      -sin(obj.w) cos(obj.w)];
                  
            obj.ideal_point = [0 0];
            obj.nadir_point = [4 4];
            obj.a = 1;
            obj.b = 10;
            obj.c = 10;
        end
        
        %% 计算目标函数
        function PopObj = CalObj(obj,X)
            X1 = X(:,1);
            X2 = X(:,2);
            
            if obj.w ~= 0
                Y  = X * obj.IRM';
                X1 = Y(:,1);
                X2 = Y(:,2);
            end
            
            a = obj.a;
            b = obj.b;
            c = obj.c;
            
            t1_hat = sign(X1).*ceil((abs(X1)-a-c/2)./(2*a+c));
            t2_hat = sign(X2).*ceil((abs(X2)-b/2)./b);
            
            one = ones(size(X1));
            
            t1 = sign(t1_hat).*min(abs(t1_hat),one);
            t2 = sign(t2_hat).*min(abs(t2_hat),one);
            
            p1 = X1 - t1*c;
            p2 = X2 - t2*b;
            
            f1 = (p1 + a).^2 + p2.^2;
            f2 = (p1 - a).^2 + p2.^2;
            
            PopObj = [f1 f2];
        end
    end
end