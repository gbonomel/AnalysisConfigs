from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nBtagMin, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow
from workflow import ttbarBackgroundProcessor

import custom_cut_functions
import custom_cuts
import custom_weights
from custom_cut_functions import *
from custom_cuts import *
from custom_weights import SF_top_pt
from params.axis_settings import axis_settings

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/lepton_scale_factors.yaml",
                                                  f"{localdir}/params/btagging.yaml",
                                                  f"{localdir}/params/btagSF_calibration.yaml",
                                                  f"{localdir}/params/plotting_style.yaml",
                                                  update=True)

samples_top = ["TTbbSemiLeptonic", "TTToSemiLeptonic", "TTTo2L2Nu"]

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_local.json",
                  f"{localdir}/datasets/signal_ttHTobb_ttToSemiLep_local.json",
                  f"{localdir}/datasets/backgrounds_MC_TTbb_local.json",
                  f"{localdir}/datasets/backgrounds_MC_ttbar_local.json",
                  f"{localdir}/datasets/backgrounds_MC_ST_WJets_local.json",
                  f"{localdir}/datasets/backgrounds_MC_ZJets.json",
                  f"{localdir}/datasets/backgrounds_MC_TTV.json",
                  f"{localdir}/datasets/backgrounds_MC_VV.json",
                  f"{localdir}/datasets/DATA_SingleEle_local.json",
                  f"{localdir}/datasets/DATA_SingleMuon_local.json",
                  ],
        "filter" : {
            "samples": ["ttHTobb",
                        "ttHTobb_ttToSemiLep",
                        "TTbbSemiLeptonic",
                        "TTToSemiLeptonic",
                        "TTTo2L2Nu",
                        "SingleTop",
                        "WJetsToLNu_HT",
                        "DYJetsToLL",
                        "VV",
                        "TTV",
                        "DATA_SingleEle",
                        "DATA_SingleMuon"
                        ],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     ] #All the years
        },
        "subsamples": {
            'DATA_SingleEle'  : {
                'DATA_SingleEle' : [get_HLTsel(primaryDatasets=["SingleEle"])]
            },
            'DATA_SingleMuon' : {
                'DATA_SingleMuon' : [get_HLTsel(primaryDatasets=["SingleMuon"]),
                                     get_HLTsel(primaryDatasets=["SingleEle"], invert=True)]
            },
            'TTbbSemiLeptonic' : {
                'TTbbSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                'TTbbSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                'TTbbSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
            'TTToSemiLeptonic' : {
                'TTToSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                'TTToSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                'TTToSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
        }
    },

    workflow = ttbarBackgroundProcessor,
    workflow_options = {"parton_jet_min_dR": 0.3,
                        "samples_top": samples_top},
    
    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(2, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [semileptonic_presel_2b],
    categories = {
        "semilep": [passthrough],
    },
    weights_classes = common_weights + [SF_top_pt],
    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                "sf_btag", "sf_btag_calib",
                "sf_jet_puId", "sf_top_pt"
            ],
            "bycategory": {}
        },
    },
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    "pileup",
                    "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                    "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                    "sf_btag", "sf_btag_calib",
                    "sf_jet_puId", "sf_top_pt"
                ],
                "bycategory": {}
            },
        },
    },
    
    variables = {
        **count_hist(name="nJets", coll="JetGood",bins=10, start=4, stop=14),
        **count_hist(name="nBJets", coll="BJetGood",bins=10, start=0, stop=10),
        **ele_hists(axis_settings=axis_settings),
        **muon_hists(axis_settings=axis_settings),
        **met_hists(coll="MET", axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=0, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=1, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=2, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=3, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=4, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=0, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=1, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=2, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=3, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=4, axis_settings=axis_settings),

        "jets_Ht" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", bins=100, start=0, stop=2500,
                label="Jets $H_T$ [GeV]")]
        ),
        "bjets_Ht" : HistConf(
          [Axis(coll="events", field="BJetGood_Ht", bins=100, start=0, stop=2500,
                label="B-Jets $H_T$ [GeV]")]
        ),
        "lightjets_Ht" : HistConf(
          [Axis(coll="events", field="LightJetGood_Ht", bins=100, start=0, stop=2500,
                label="Light-Jets $H_T$ [GeV]")]
        ),
        "jets_Ht_coarsebins" : HistConf(
            [Axis(coll="events", field="JetGood_Ht", bins=15, start=100, stop=1600,
                  label="Jets $H_T$ [GeV]")]
        ),
        "bjets_Ht_coarsebins" : HistConf(
            [Axis(coll="events", field="BJetGood_Ht", bins=15, start=100, stop=1600,
                  label="B-Jets $H_T$ [GeV]")]
        ),
        "lightjets_Ht_coarsebins" : HistConf(
            [Axis(coll="events", field="LightJetGood_Ht", bins=15, start=100, stop=1600,
                  label="Light-Jets $H_T$ [GeV]")]
        ),
        "deltaRbb_min" : HistConf(
            [Axis(coll="events", field="deltaRbb_min", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}$")]
        ),
        "deltaEtabb_min" : HistConf(
            [Axis(coll="events", field="deltaEtabb_min", bins=50, start=0, stop=5,
                  label="$\Delta \eta_{bb}$")]
        ),
        "deltaPhibb_min" : HistConf(
            [Axis(coll="events", field="deltaPhibb_min", bins=50, start=0, stop=5,
                  label="$\Delta \phi_{bb}$")]
        ),
        "mbb_closest" : HistConf(
            [Axis(coll="events", field="mbb_closest", bins=50, start=0, stop=500,
                    label="$m_{bb}(min \Delta R(bb))$ [GeV]")]
        ),
        "mbb_min" : HistConf(
            [Axis(coll="events", field="mbb_min", bins=50, start=0, stop=500,
                    label="$m_{bb}^{min}$ [GeV]")]
        ),
        "mbb_max" : HistConf(
            [Axis(coll="events", field="mbb_max", bins=50, start=0, stop=500,
                    label="$m_{bb}^{max}$ [GeV]")]
        ),
        "deltaRbb_avg" : HistConf(
            [Axis(coll="events", field="deltaRbb_avg", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}^{avg}$")]
        ),
        # 2D plots
        "Njet_Ht": HistConf(
            [
                Axis(coll="events", field="nJetGood",bins=[4,5,6,7,8,9,11,20],
                     type="variable",   label="$N_{JetGood}$"),
                Axis(coll="events", field="JetGood_Ht",
                     bins=[0,100,200,300,400,500,750,1000,1250,1500,2000,2500,5000],
                     type="variable",
                     label="Jets $H_T$ [GeV]"),
            ]
        ),
        "Njet_Ht_finerbins": HistConf(
            [
                Axis(coll="events", field="nJetGood",bins=[4,5,6,7,8,9,11,20],
                     type="variable",   label="$N_{JetGood}$"),
                Axis(coll="events", field="JetGood_Ht",
                     bins=[0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,5000],
                     type="variable",
                     label="Jets $H_T$ [GeV]"),
            ]
        ),
    },
    columns = {
        "common": {
            "inclusive": [],
            "bycategory": {}
        },
        #"bysample": {
        #    sample : columns for sample in samples_top
        #}
    }
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
cloudpickle.register_pickle_by_value(custom_weights)
