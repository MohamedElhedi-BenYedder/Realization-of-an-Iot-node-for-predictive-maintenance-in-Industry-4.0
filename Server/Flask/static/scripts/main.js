"use strict";
exports.__esModule = true;
//code here
var baseURL = "http://localhost:5000";

var ConnectedDevices = document.querySelector("#Connected-Devices");
var ConnectedDevicesURL = baseURL + "/engine/ConnectedDevices";
var ConnectedDevicesdata = fetch(ConnectedDevicesURL).then(function (response) {
    return response.json().then(function (ConnectedDevicesdata) {
        ConnectedDevices.innerHTML += ConnectedDevicesdata;
        console.log(ConnectedDevicesdata);
        }).catch(function (err) {
        console.log("loading error!");
    });
});


var engineCycle = document.querySelector("#engineCycle");
var engineCycledataURL = baseURL + "/DW/data";
var engineCycledata = fetch(engineCycledataURL).then(function (response) {
    return response.json().then(function (engineCycledata) {
        engineCycle.innerHTML += engineCycledata+`</div>`;
        console.log(engineCycledata);
    }).catch(function (err) {
        console.log("loading error!");
    });
});

var columnstotreat  = document.querySelector("#columnstotreat");
var columnstotreatURL = baseURL + "/DW/columns_to_treat";
var fxURL = baseURL + "/DW/df_fx";
var periodURL = baseURL + "/DW/period";
var period= fetch(periodURL).then(function (response) {
    return response.json().then(function (period) {
        columnstotreat.innerHTML += period +` ):`;
        console.log(columnstotreatdata);
    }).catch(function (err) {
        console.log("loading error!");
    });
});
var columnstotreatdata = fetch(columnstotreatURL).then(function (response) {
    return response.json().then(function (columnstotreatdata) {
        columnstotreat.innerHTML += columnstotreatdata+`</div>=> So we have as result :`;
        console.log(columnstotreatdata);
    }).catch(function (err) {
        console.log("loading error!");
    });
});
var fxtable = document.querySelector("#fx_tab");
var fxdata = fetch(fxURL).then(function (response) {
    return response.json().then(function (fxdata) {
        fxtable.innerHTML += fxdata;
        console.log(fxdata);
    }).catch(function (err) {
        console.log("loading error!");
    });
});

var ttfdoc = document.querySelector("#ttftab");
var ttfURL = baseURL + "/MS/ttf";
var ttfdata = fetch(ttfURL).then(function (response) {
    return response.json().then(function (ttfdata) {
        ttfdoc.innerHTML += ttfdata+`</div>`;
        console.log(ttfdata);
        }).catch(function (err) {
        console.log("loading error!");
    });
});

var bncdoc = document.querySelector("#bnctab");
var bncURL = baseURL + "/MS/bnc";
var bncdata = fetch(bncURL).then(function (response) {
    return response.json().then(function (bncdata) {
        bncdoc.innerHTML += bncdata+`</div>`;
        console.log(bncdata);
        }).catch(function (err) {
        console.log("loading error!");
    });
});

var mccdoc = document.querySelector("#mcctab");
var mccURL = baseURL + "/MS/mcc";
var mccdata = fetch(mccURL).then(function (response) {
    return response.json().then(function (mccdata) {
        mccdoc.innerHTML += mccdata+`</div>`;
        console.log(mccdata);
        }).catch(function (err) {
        console.log("loading error!");
    });
});

var smdoc = document.querySelector("#sm");
var smURL = baseURL + "/MS/selectedModels";
var smdata = fetch(smURL).then(function (response) {
    return response.json().then(function (smdata) {
        smdoc.innerHTML += smdata;
        console.log(smdata);
        }).catch(function (err) {
        console.log("loading error!");
    });
});
var resdoc = document.querySelector("#res");
var resURL = baseURL + "/MS/predicitions";
var resdata = fetch(resURL).then(function (response) {
    return response.json().then(function (resdata) {
        resdoc.innerHTML += resdata+`</div>`;
        console.log(resdata);
        }).catch(function (err) {
        console.log("loading error!");
    });
});

