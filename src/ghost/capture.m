function OK = capture(FrameTime, AcqTime, filename)
    addpath('C:\Program Files\Photon Force\Matlab\wrapper')
    % clearvars -GLOBAL series hax
    % global AcqTime FrameNum FrameTime MotorPos series seriesArray 
    % global tauRange CCwindow CCframetot frame leftReg rightReg CCframe
    % global hax % #ok<NUSED>
    % initialize PF32

    OK = pf_getLibraryStatus;
    if(OK)
        % Close the library if it is already open. The scripts will load it
        % when needed.
        pf_close
    end
    [pf32, alias, report] = pf_open;
    % pf_setMode(pf32, 'photon_counting');
    % calllib(alias, 'setEnablePositionalData', pf32, 0);
    % calllib(alias, 'purgeBulkFrameBuffer', pf32);

    % pf_setNumberAccumulations(pf32, 1);
    % pf_setExposure(pf32, FrameTime);
    % pf_getLibraryStatus;

    FrameNum = 5000
    series = pf_getMultipleFrames(pf32, FrameNum);
    series = rot90(series,3);
    img = sum(series,3);
    filename
    writematrix(img, filename)

    % pf_close()
    OK =0
end
