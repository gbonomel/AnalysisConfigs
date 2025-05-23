import awkward as ak
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor
from pocket_coffea.lib.objects import btagging
from pocket_coffea.lib.deltaR_matching import metric_eta, metric_phi
from pocket_coffea.lib.deltaR_matching import object_matching
from pocket_coffea.lib.parton_provenance import get_partons_provenance_ttHbb, get_partons_provenance_ttbb4F, get_partons_provenance_tt5F
from custom_weights import get_sf_top_pt

class ttbarBackgroundProcessor(ttHbbBaseProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        self.dr_min = self.workflow_options["parton_jet_min_dR"]

    @classmethod
    def available_variations(cls):
        vars = super().available_variations()
        variations_sf_ele_trigger = ["stat", "pileup", "era", "ht"]
        available_sf_ele_trigger_variations = [f"sf_ele_trigger_{v}" for v in variations_sf_ele_trigger]
        variations_sf_btag = ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2", "cferr1", "cferr2"]
        available_sf_btag_variations = [f"sf_btag_{v}" for v in variations_sf_btag]
        vars = vars + available_sf_ele_trigger_variations + available_sf_btag_variations
        return vars

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation=variation)

        self.events["LightJetGood"] = btagging(
            self.events["JetGood"],
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet["btag"]["wp"],
            veto=True
        )

    def define_common_variables_before_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)

        # Compute the scalar sum of the transverse momenta of the b-jets and light jets
        self.events["BJetGood_Ht"] = ak.sum(abs(self.events.BJetGood.pt), axis=1)
        self.events["LightJetGood_Ht"] = ak.sum(abs(self.events.LightJetGood.pt), axis=1)

    def define_common_variables_after_presel(self, variation):
        super().define_common_variables_before_presel(variation=variation)

        # Compute the `is_electron` flag for LeptonGood
        self.events["LeptonGood"] = ak.with_field(
            self.events.LeptonGood,
            ak.values_astype(self.events.LeptonGood.pdgId == 11, bool),
            "is_electron"
        )

        # Compute deltaR(b, b) of all possible b-jet pairs.
        # We require deltaR > 0 to exclude the deltaR between the jets with themselves
        deltaR = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"]), axis=2)
        deltaEta = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_eta), axis=2)
        deltaPhi = ak.flatten(self.events["BJetGood"].metric_table(self.events["BJetGood"], metric=metric_phi), axis=2)
        deltaR = deltaR[deltaR > 0.]
        deltaEta = deltaEta[deltaEta > 0.]
        deltaPhi = deltaPhi[deltaPhi > 0.]

        # Get the deltaR with no possibility of repetition of identical b-jet pairs

        # Get all the possible combinations of b-jet pairs
        pairs = ak.argcombinations(self.events["BJetGood"], 2, axis=1)
        b1 = self.events.BJetGood[pairs.slot0]
        b2 = self.events.BJetGood[pairs.slot1]

        # Compute deltaR between the pairs
        deltaR_unique = b1.delta_r(b2)
        idx_pairs_sorted = ak.argsort(deltaR_unique, axis=1)
        pairs_sorted = pairs[idx_pairs_sorted]

        # Compute the minimum deltaR(b, b), deltaEta(b, b), deltaPhi(b, b)
        self.events["deltaRbb_min"] = ak.min(deltaR, axis=1)
        self.events["deltaEtabb_min"] = ak.min(deltaEta, axis=1)
        self.events["deltaPhibb_min"] = ak.min(deltaPhi, axis=1)

        # Compute the invariant mass of the closest b-jet pair, the minimum and maximum invariant mass of all b-jet pairs
        mbb = (self.events.BJetGood[pairs_sorted.slot0] + self.events.BJetGood[pairs_sorted.slot1]).mass
        ptbb = (self.events.BJetGood[pairs_sorted.slot0] + self.events.BJetGood[pairs_sorted.slot1]).pt
        htbb = self.events.BJetGood[pairs_sorted.slot0].pt + self.events.BJetGood[pairs_sorted.slot1].pt
        self.events["mbb_closest"] = mbb[:,0]
        self.events["mbb_min"] = ak.min(mbb, axis=1)
        self.events["mbb_max"] = ak.max(mbb, axis=1)
        self.events["deltaRbb_avg"] = ak.mean(deltaR_unique, axis=1)
        self.events["ptbb_closest"] = ptbb[:,0]
        self.events["htbb_closest"] = htbb[:,0]

        # Save top and anti-top pT
        samples_top = self.workflow_options["samples_top"]
        if self._isMC & (self._sample in samples_top):
            mask_tops = ~ak.is_none(self.events.GenPart.parent, axis=1) & self.events.GenPart.hasFlags("isLastCopy") & (abs(self.events.GenPart.pdgId) == 6)
            part = self.events.GenPart[mask_tops]
            assert all(ak.num(part.pt, axis=1) == 2), f"There should be exactly 2 tops in the event. Please check the mask to select the top and anti-top GenParticles. Sample: {self._sample}"
            top = part[part.pdgId == 6][:,0]
            antitop = part[part.pdgId == -6][:,0]
            self.events["top_pt"] = top.pt
            self.events["antitop_pt"] = antitop.pt
            self.events["sf_top_pt"] = get_sf_top_pt(self.events, self.events.metadata)

        # Define labels for btagged jets at different working points
        for wp, val in self.params.btagging.working_point[self._year]["btagging_WP"].items():
            self.events["JetGood"] = ak.with_field(
                self.events.JetGood,
                ak.values_astype(self.events.JetGood[self.params.btagging.working_point[self._year]["btagging_algorithm"]] > val, int),
                f"btag_{wp}"
            )

    def do_parton_matching(self) -> ak.Array:
        # Selects quarks at LHE level
        isOutgoing = self.events.LHEPart.status == 1
        isParton = (abs(self.events.LHEPart.pdgId) < 6) | (self.events.LHEPart.pdgId == 21)
        quarks = self.events.LHEPart[isOutgoing & isParton]

        # Select b-quarks at Gen level, coming from H->bb decay
        if self._sample in ['ttHTobb', 'ttHTobb_ttToSemiLep']:
            higgs = self.events.GenPart[
                (self.events.GenPart.pdgId == 25)
                & (self.events.GenPart.hasFlags(['fromHardProcess']))
            ]
            higgs = higgs[ak.num(higgs.childrenIdxG, axis=2) == 2]
            higgs_partons = ak.with_field(
                ak.flatten(higgs.children, axis=2), 25, "from_part"
            )
            # DO NOT sort b-quarks by pt
            # if not we are not able to match them with the provenance
            quarks = ak.with_name(
                ak.concatenate((quarks, higgs_partons), axis=1),
                name='PtEtaPhiMCandidate',
            )

        # Get the interpretation
        if self._sample in ['ttHTobb', 'ttHTobb_ttToSemiLep']:
            prov = get_partons_provenance_ttHbb(
                ak.Array(quarks.pdgId, behavior={}), ak.ArrayBuilder()
            ).snapshot()
            self.events["HiggsParton"] = self.events.LHEPart[
                self.events.LHEPart.pdgId == 25
            ]
        elif self._sample == "TTbbSemiLeptonic":
            prov = get_partons_provenance_ttbb4F(
                ak.Array(quarks.pdgId, behavior={}), ak.ArrayBuilder()
            ).snapshot()
        elif self._sample == "TTToSemiLeptonic":
            prov = get_partons_provenance_tt5F(
                ak.Array(quarks.pdgId, behavior={}), ak.ArrayBuilder()
            ).snapshot()
        else:
            prov = -1 * ak.ones_like(quarks)

        # Adding the provenance to the quark object
        quarks = ak.with_field(quarks, prov, "provenance")

        # Calling our general object_matching function.
        # The output is an awkward array with the shape of the second argument and None where there is no matching.
        # So, calling like this, we will get out an array of matched_quarks with the dimension of the JetGood.
        matched_quarks, matched_jets, deltaR_matched = object_matching(
            quarks, self.events.JetGood, dr_min=self.dr_min
        )

        # Saving leptons and neutrino parton level
        self.events["LeptonParton"] = self.events.LHEPart[
            (self.events.LHEPart.status == 1)
            & (abs(self.events.LHEPart.pdgId) > 10)
            & (abs(self.events.LHEPart.pdgId) < 15)
        ]

        self.events["Parton"] = quarks
        self.events["PartonMatched"] = ak.with_field(
            matched_quarks, deltaR_matched, "dRMatchedJet"
        )
        self.events["JetGoodMatched"] = ak.with_field(
            matched_jets, deltaR_matched, "dRMatchedJet"
        )
        self.matched_partons_mask = ~ak.is_none(self.events.JetGoodMatched, axis=1)

    def count_partons(self):
        self.events["nParton"] = ak.num(self.events.Parton, axis=1)
        self.events["nPartonMatched"] = ak.count(
            self.events.PartonMatched.pt, axis=1
        )  # use count since we have None

    def process_extra_after_presel(self, variation) -> ak.Array:
        if self._isMC & (self._sample in ["ttHTobb", "ttHTobb_ttToSemiLep", "TTbbSemiLeptonic", "TTToSemiLeptonic"]):
            self.do_parton_matching()
            self.count_partons()
