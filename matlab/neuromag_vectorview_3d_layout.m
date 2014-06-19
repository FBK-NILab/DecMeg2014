data= load('../DecMeg2014-master/additional_files/NeuroMagSensorsDeviceSpace.mat');
    position = data.pos;
    orientation = data.ori;
    label = data.lab;
    sensor_type = data.typ;

    %Normalize orientation for visualization purpose:
    total=orientation.*orientation;
    orientation = orientation ./ repmat(sqrt(sum(total,2)),1,3);
    quiver3(position(:,1), position(:,2), position(:,3), orientation(:,1), orientation(:,2), orientation(:,3));
