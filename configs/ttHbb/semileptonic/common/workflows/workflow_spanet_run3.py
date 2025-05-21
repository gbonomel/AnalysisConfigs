import os
import sys
import awkward as ak
from dask.distributed import get_worker

from .workflow_tthbb_genmatching import ttHbbPartonMatchingProcessorFull
from ..params.quantile_transformer import WeightedQuantileTransformer

import numpy as np

class SpanetInferenceProcessor(ttHbbPartonMatchingProcessorFull):
    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)
        if not "spanet_model" in self.workflow_options:
            raise ValueError("Key `spanet_model` not found in workflow options. Please specify the path to the ONNX model.")
        elif not self.workflow_options["spanet_model"].endswith(".onnx"):
            raise ValueError("Key `spanet_model` should be the path of an ONNX model.")


    def process_extra_after_presel(self, variation) -> ak.Array:
        super().process_extra_after_presel(variation)

        try:
            worker = get_worker()
        except ValueError:
            worker = None

        if worker is None:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            #sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            model_session = ort.InferenceSession(
                self.workflow_options["spanet_model"],
                sess_options = sess_options,
                providers=['CPUExecutionProvider']
            )
        else:
            model_session = worker.data['model_session_spanet']

        #for input in model_session.get_inputs():
        #    print(f"{input.name}, {input.shape}")

        #for output in model_session.get_outputs():
        #    print(f"{output.name}, {output.shape}")

        btagging_algorithm = self.params.btagging.working_point[self._year]["btagging_algorithm"]
        pad_dict = {btagging_algorithm:0., "btag_L":0, "btag_M":0, "btag_T":0, "btag_XT":0, "btag_XXT":0, "pt":0., "phi":0., "eta":0.} 
        jets_padded = ak.zip(
            {key : ak.fill_none(ak.pad_none(self.events.JetGood[key], 16, clip=True), value) for key, value in pad_dict.items()}
        )
        
        data = np.transpose(
            np.stack([
                np.log(1 + ak.to_numpy(jets_padded.pt)),
                ak.to_numpy(jets_padded.eta),
                ak.to_numpy(jets_padded.phi),
                ak.to_numpy(jets_padded[btagging_algorithm]),
                ak.to_numpy(jets_padded.btag_L),
                ak.to_numpy(jets_padded.btag_M),
                ak.to_numpy(jets_padded.btag_T),
                ak.to_numpy(jets_padded.btag_XT),
                ak.to_numpy(jets_padded.btag_XXT)
            ]),
            axes=[1,2,0]).astype(np.float32)

        mask = ~ak.to_numpy(jets_padded.pt == 0)

        met_data = np.stack([np.log(1+ ak.to_numpy(self.events.MET.pt)),
                             ak.zeros_like(self.events.MET.pt).to_numpy(),
                             #ak.to_numpy(self.events.MET.eta), # eta not included 
                             ak.to_numpy(self.events.MET.phi)
                             ], axis=1)[:,None,:].astype(np.float32)

        lep_data = np.stack([np.log(1 + ak.to_numpy(self.events.LeptonGood[:,0].pt)),
                             ak.to_numpy(self.events.LeptonGood[:,0].eta),
                             ak.to_numpy(self.events.LeptonGood[:,0].phi),
                             ], axis=1)[:,None,:].astype(np.float32)

        ht_array = ak.to_numpy(self.events.JetGood_Ht[:,None, None]).astype(np.float32) # not log normalized in our case
        #ht_array = ak.to_numpy(np.log(self.events.JetGood_Ht[:,None, None])).astype(np.float32) # if log normalized
        
        mask_global = np.ones(shape=[met_data.shape[0], 1]) == 1
        output_names = ["EVENT/signal"] #["EVENT/tthbb", "EVENT/ttbb", "EVENT/ttcc", "EVENT/ttlf"]
        
        outputs = model_session.run(input_feed={
            "Jet_data": data,
            "Jet_mask": mask,
            "Met_data": met_data,
            "Met_mask": mask_global,
            "Lepton_data": lep_data,
            "Lepton_mask": mask_global,
            "Event_data": ht_array, 
            "Event_mask": mask_global},
        output_names=output_names
        )
        # creates output branch "spanet_output" containing all the samples included in the multiclassifier
        outputs_zipped = dict(zip(output_names, outputs))
        self.events["spanet_output"] = ak.zip(
            {
                key.split("/")[-1]: ak.from_numpy(value[:,:]) for key, value in outputs_zipped.items()
            }
        )
        # define variable for each sample used in the classifier 
        self.events["spanet_tthbb"]     = self.events.spanet_output.signal[:,1]
        self.events["spanet_ttbb"]      = self.events.spanet_output.signal[:,2]
        self.events["spanet_ttcc"]      = self.events.spanet_output.signal[:,3]
        self.events["spanet_ttlf"]      = self.events.spanet_output.signal[:,4]
        self.events["spanet_ttv"]       = self.events.spanet_output.signal[:,5]
        self.events["spanet_singletop"] = self.events.spanet_output.signal[:,6]

        # Transform ttHbb score with quantile transformation
        params_quantile_transformer = self.params["quantile_transformer"][self.events.metadata["year"]]
        transformer = WeightedQuantileTransformer(n_quantiles=params_quantile_transformer["n_quantiles"], output_distribution=params_quantile_transformer["output_distribution"])
        transformer.load(params_quantile_transformer["file"])
        transformed_score = transformer.transform(self.events.spanet_tthbb)
        self.events["spanet_tthbb_transformed"] = transformed_score