function OK = capture(FrameTime, AcqTime)
    disp("capture!")

    OK = pf_getLibraryStatus;
    if(OK)
        % Close the library if it is already open. The scripts will load it
        % when needed.
        pf_close
    end

    [pf32, alias, report] = pf_open;
    pf_setMode(pf32, 'TCSPC_sys_master');
    pf_setNumberAccumulations(pf32, 1);
    pf_setExposure(pf32, FrameTime);

    FrameNum = int32((AcqTime/1000)/(FrameTime*1e-6));
    img = pf_getMultipleFrames(pf32, FrameNum);
    series = transpose(reshape(img,[32*32,FrameNum]));

    hist = sum(series ~= 0, 3, "default");
    pf_close(pf32);
    
    % % capture the image
    
    OK = 0
end