import yaml
import numpy as np
import awkward as ak
import correctionlib
from pocket_coffea.lib.weights.weights import WeightLambda, WeightWrapper, WeightData
#from pocket_coffea.lib.scale_factors import sf_L1prefiring

samples_top = ["TTbbSemiLeptonic", "TTToSemiLeptonic", "TTTo2L2Nu","TTTo2L2Nu","TTToLNu2Q"]

SF_top_pt = WeightLambda.wrap_func(
    name="sf_top_pt",
    function=lambda params, metadata, events, size, shape_variations:
            get_sf_top_pt(events, metadata),
    has_variations=False
    )

def get_sf_top_pt(events, metadata):
    if metadata["sample"] in samples_top:
        #print("Computing top pt reweighting for sample: ", metadata["sample"])
        part = events.GenPart
        part = part[~ak.is_none(part.parent, axis=1)]
        part = part[part.hasFlags("isLastCopy")]
        part = part[abs(part.pdgId) == 6]
        part = part[ak.argsort(part.pdgId, ascending=False)]

        arg = {
            "a": 0.103,
            "b": -0.0118, 
            "c": -0.000134,
            "d": 0.973
        }
        top_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,0]) + arg["c"] * part.pt[:,0] + arg["d"]
        antitop_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,1]) + arg["c"] * part.pt[:,1] + arg["d"]
        weight = np.sqrt(ak.prod([top_weight, antitop_weight], axis=0))
        # for i in range(10):
            # print("Top pt: {},   Top SF: {},   AntiTop pt :  {},   AntiTop SF: {}".format(part.pt[i,0], top_weight[i], part.pt[i,1], antitop_weight[i]))
        return weight#, np.zeros(np.shape(weight)), ak.copy(weight)
    else:
        return np.ones(len(events), dtype=np.float64)

def sf_ttlf_calib(params, sample, year, njets, jetsHt):
    '''Correction to tt+LF background computed by correcting tt+LF to data minus the other backgrounds in 2D:
    njets-JetsHT bins. Each year has a different correction stored in the correctionlib format.'''
    assert sample == "TTToSemiLeptonic", "This weight is only for TTToSemiLeptonic sample"
    cset = correctionlib.CorrectionSet.from_file(
        params.ttlf_calibration[year]["file"]
    )
    corr = cset[params.ttlf_calibration[year]["name"]]
    w = corr.evaluate(ak.to_numpy(njets), ak.to_numpy(jetsHt))
    return w

def get_njet_reweighting(events, reweighting_map_njet, mask=None):
    njet = ak.num(events.JetGood)
    w_nj = np.ones(len(events))
    if mask is None:
        mask = np.ones(len(events), dtype=bool)
    for nj in range(4,7):
        mask_nj = (njet == nj)
        w_nj = np.where(mask & mask_nj, reweighting_map_njet[nj], w_nj)
    for nj in range(7,21):
        w_nj = np.where(mask & (njet >= 7), reweighting_map_njet[nj], w_nj)
    return w_nj

DCTR_weight = WeightLambda.wrap_func(
    name="dctr_weight",
    function=lambda params, metadata, events, size, shape_variations:
            events.dctr_output.weight,
    has_variations=False
    )

class SF_ttlf_calib(WeightWrapper):
    name = "sf_ttlf_calib"
    has_variations = False

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = "JetGood"

    def compute(self, events, size, shape_variation):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        out = sf_ttlf_calib(self._params,
                            sample=self._metadata["sample"],
                            year=self._metadata["year"],
                            # Assuming n*JetCollection* is defined
                            njets=events[f"n{self.jet_coll}"],
                            jetsHt=jetsHt
                            )
        return WeightData(
            name = self.name,
            nominal = out, #out[0] only if has_variations = True
            #up = out[1],
            #down = out[2]
            )

class SF_njet_reweighting(WeightWrapper):
    '''Correction to tt+bb background computed to match data/MC in the number of jets.
    The corection applied during training of the DCTR model is stored in a yaml file.'''
    name = "sf_njet_reweighting"
    has_variations = False

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        assert metadata["sample"] == "TTbbSemiLeptonic", "This weight is only for TTbbSemiLeptonic sample"
        params_njet_reweighting = params.dctr["njet_reweighting"]
        file, key = params_njet_reweighting["file"], params_njet_reweighting["key"]
        with open(file, "r") as f:
            self.reweighting_map_njet = yaml.safe_load(f)[key]

    def compute(self, events, size, shape_variation):
        out = get_njet_reweighting(events, self.reweighting_map_njet)
        return WeightData(
            name = self.name,
            nominal = out
        )

class SF_LHE_pdf_weight(WeightWrapper):
    '''Weight corresponding to the LHEPdfWeight in the LHE event.
    The up and down variations are computed as the sum in quadrature of the differences
    the alternative weights with respect to the nominal weight.'''
    name = "sf_lhe_pdf_weight"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)

    def compute(self, events, size, shape_variation):
        w = events.LHEPdfWeight.to_numpy()
        w_nom = np.ones(len(events))
        #assert all(w_nom == 1), "The nominal weight is not 1."
        dw = np.sqrt(np.sum((w - 1.0) ** 2, axis=1))
        w_up = 1.0 + dw
        w_down = 1.0 - dw
        return WeightData(
            name = self.name,
            nominal = w_nom,
            up = w_up,
            down = w_down
        )

#SF_L1prefiring = WeightLambda.wrap_func(
#    name="sf_L1prefiring",
#    function=lambda params, metadata, events, size, shape_variations:
#        sf_L1prefiring(events) if metadata["year"] in ["2016_PreVFP", "2016_PostVFP", "2017"] else (np.ones(len(events)), np.ones(len(events)), np.ones(len(events))),
#    has_variations=True
#    )

class SF_btag_fixed_wp(WeightWrapper):
    name = "sf_btag_fixed_wp"
    has_variations = True
    
    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.params = params
        self.metadata = metadata
        self._variations = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["comb"] #+ params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][metadata["year"]]["light"]
        
    def compute(self, events, size, shape_variation):
        
        if shape_variation == "nominal":
            nominal, variations, up_var, down_var = get_sf_btag_fixed_wp(
                self.params, 
                events.JetGood, 
                self.metadata["year"], 
                self.metadata["sample"], 
                return_variations=True
            )
            return WeightDataMultiVariation(
                name = self.name,
                nominal = nominal,
                variations = variations,
                up = up_var,
                down = down_var
            )
        else:
            return WeightData(
                name = self.name, 
                nominal = get_sf_btag_fixed_wp(
                    self.params, 
                    events.JetGood, 
                    self.metadata["year"], 
                    self.metadata["sample"], 
                    return_variations=False
                )
            )

def get_sf_btag_fixed_wp(params, Jets, year, sample, return_variations=True):

    sampleGroups = {
        "ttH" :  [
            "TTH_Hto2B",
            "TTHtoNon2B",
        ],
        "ttbar" :  [
            "TTTo2L2Nu",
            "TTToLNu2Q",
        ],
        "top"  :   [
            "TTLL_ML_to50",
            "TTLL_ML_50",
            "TTNuNu",
            "TTLNu",
            "TTZ",
            "TWminus",
            "TWplus",
        ],
        "Vjets" : [
            "WJetsToLNu",
            "ZZ",
            "WW",
            "WZ",
            "DYJetsToLL",
        ]
    }

    btag_effi_sample_group = ""
    for sampleGroupName, sampleGroup in sampleGroups.items():
        if sample in sampleGroup:
            btag_effi_sample_group = sampleGroupName
    if btag_effi_sample_group == "":
        print("WARNING: Sample does not correspond to one of the given sample groupings!")
    
    paramsBtagSf = params["jet_scale_factors"]["btagSF"][year]
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"]
    sf_file = paramsBtagSf["file"]
    btag_effi_file = paramsBtagSf["btagEfficiencyFile"][btag_algo]
    # Jets = events["JetGood"]

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(btag_effi_file)
    btag_sf_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])

    jetpt_heavy = ak.flatten(Jets["pt"][Jets["hadronFlavour"]>3])
    jeteta_heavy = ak.flatten(Jets["eta"][Jets["hadronFlavour"]>3])
    jetflav_heavy = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]>3])

    jetpt_light = ak.flatten(Jets["pt"][Jets["hadronFlavour"]<4])
    jeteta_light = ak.flatten(Jets["eta"][Jets["hadronFlavour"]<4])
    jetflav_light = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]<4])

    wp = params.object_preselection["Jet"]["btag"]["wp"]
    print(list(btag_effi_corr_set.keys()))
    print(btag_effi_sample_group + "_wp_" + wp)
    effi_MC = ak.unflatten(
        btag_effi_corr_set[btag_effi_sample_group + "_wp_" + wp].evaluate(jetpt, jeteta, jetflav), 
        counts=jetcounts
    )

    if return_variations:
        paramsBtagVar = params["systematic_variations"]["weight_variations"]["sf_btag_fixed_wp"][year]
        #sf_btag_collection = paramsBtagVar["sf_collection_to_use"]
        #sf_btag_lightFlavorName = paramsBtagVar["light_flavor_name"]
        #if sf_btag_collection not in ["comb", "mujets"]:
        #    print(
        #        "ERROR: type_sf_btag has to be either comb or mujets, "+\
        #        "see documentation here https://btv-wiki.docs.cern.ch"+\
        #        "/PerformanceCalibration/SFUncertaintiesAndCorrelations"+\
        #        "/#working-point-based-sfs-fixedwp-sfs")
        
        btag_sf_name = paramsBtagSf["fixed_wp_name_map"][btag_algo]

        heavy_variations = paramsBtagVar["comb"]
        light_variations = paramsBtagVar["light"]

        variation_names = []
        if heavy_variations == None:
            heavyVarUp,  heavyVarDown = [], []
        if len(heavy_variations)<=1:
            heavyVarUp = ["up"]
            heavyVarDown = ["down"]
            variation_names.append("heavyFlavor")
        else: 
            heavyVarUp = ["up_"+var for var in heavy_variations]
            heavyVarDown = ["down_"+var for var in heavy_variations]
            variation_names += list(heavy_variations)

        if light_variations == None:
            lightVarUp,  lightVarDown = [], []
        if len(light_variations)<=1:
            lightVarUp = ["up"]
            lightVarDown = ["down"]
            variation_names.append("lightFlavor")
        else: 
            lightVarUp = ["up_"+var for var in light_variations]
            lightVarDown = ["down_"+var for var in light_variations]
            variation_names += ["lightFlavor_"+var for var in light_variations]
            
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            },
            "up": {
                "light": ["central" for var in heavyVarUp]+lightVarUp,
                "heavy": heavyVarUp+["central" for var in lightVarUp],
                "btag_sf": []
            },
            "down": {
                "light": ["central" for var in heavyVarDown]+lightVarDown,
                "heavy": heavyVarDown+["central" for var in lightVarDown],
                "btag_sf": []
            }
        }
    else:
        variationsDict = {
            "central": {
                "light": ["central"],
                "heavy": ["central"],
                "btag_sf": []
            }
        }

    for variationType, variationColl in variationsDict.items():
        for variation_light, variation_heavy in zip(variationColl["light"], variationColl["heavy"]):
            sf_light_flat = btag_sf_corr_set[btag_sf_name+"_light"].evaluate(
                variation_light, 
                wp, 
                jetflav_light,
                abs(jeteta_light), 
                jetpt_light
            )
            sf_heavy_flat = btag_sf_corr_set[btag_sf_name+"_comb"].evaluate(
                variation_heavy, 
                wp, 
                jetflav_heavy, 
                abs(jeteta_heavy), 
                jetpt_heavy
            )
            sf_flat = ak.to_numpy(ak.copy(jetpt))
            sf_flat[jetflav>3] = sf_heavy_flat
            sf_flat[jetflav<4] = sf_light_flat
            sf_DATA_MC = ak.unflatten(sf_flat, jetcounts)
            

            effi_DATA = ak.prod([sf_DATA_MC, effi_MC], axis=0)

            Jets = ak.with_field(Jets, effi_MC, "effi_MC_"+wp)
            Jets = ak.with_field(Jets, effi_DATA, "effi_DATA_"+wp)

            Jets_not_tagged = Jets[Jets[btag_algo] <= btag_wps[wp]]
            Jets_tagged = Jets[Jets[btag_algo] > btag_wps[wp]]

            p_MC_1 = 1 - Jets_not_tagged["effi_MC_"+wp]
            p_MC_2 = Jets_tagged["effi_MC_"+wp]

            p_DATA_1 = 1 - Jets_not_tagged["effi_DATA_"+wp]
            p_DATA_2 = Jets_tagged["effi_DATA_"+wp]

            p_MC = ak.concatenate([p_MC_1, p_MC_2], axis=1)
            p_MC = ak.prod(p_MC, axis=-1)
            p_DATA = ak.concatenate([p_DATA_1, p_DATA_2], axis=1)
            p_DATA = ak.prod(p_DATA, axis=-1)

            btag_sf_fixed_wp = np.divide(p_DATA, p_MC)

            variationsDict[variationType]["btag_sf"].append(btag_sf_fixed_wp)

    if return_variations:
        return (variationsDict["central"]["btag_sf"][0], 
                variation_names, 
                variationsDict["up"]["btag_sf"], 
                variationsDict["down"]["btag_sf"])
    return variationsDict["central"]["btag_sf"][0]