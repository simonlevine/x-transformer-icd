import typing as t

import tempfile
import argparse

import numpy as np
import scipy.sparse as smat
from sklearn.preprocessing import normalize as sk_normalize
from pipeline.xbert_preprocessing import
from xbert.ranker import LinearModel
from xbert.transformer import MODEL_CLASSES, TransformerMatcher
from xbert.rf_linear import PostProcessor, HierarchicalMLModel

TOPk = 10

def embed_and_predict(discharge_summary,
                      model_class="longformer",
                      num_labels=128,
                      matcher_dir="'./data/intermediary-data/xbert_outputs/pifa-neural-s0/matcher',"):

    with tempfile.NamedTemporaryFile("wb") as f_X1:
        smat.save_npz(
            f_X1.name,
            np.array([discharge_summary]).replace(r'\n', ' ', regex=True)
        )
        f_X1.seek(0)
        X1 = HierarchicalMLModel.load_feature_matrix(f_X1.name)

    # xbert_trained_model_prediction.sh -> transformer.py
    with tempfile.NamedTemporaryFile("wb") as f_X2:
        TransformerMatcher.set_device(argparse.NameSpace(no_cuda=True))
        matcher = TransformerMatcher(num_clusters=num_labels)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_class]
        matcher.config = config_class.from_pretrained(matcher_dir, gradient_checkpointing=True) # config fix
        matcher.config.num_labels = num_labels
        matcher.config.output_hidden_states = True #FOR BERTVIZ PURPOSES?
        model = model_class.from_pretrained(matcher_dir, config=matcher.config)
        matcher.model = model.to("cpu")
        *_, C_tst_pred, tst_embeddings = matcher.predict(
            argparse.NameSpace(**{
                'model_type': model_class,
                'model_name_or_path': 'custom_models/biomed_roberta_base-4096',
                'trn_feat_path': './data/intermediary-data/xbert_outputs/proc_data/X.trn.tomodel.pkl',
                'tst_feat_path': './data/intermediary-data/xbert_outputs/proc_data/X.tst.tomodel.pkl',
                'trn_label_path': './data/intermediary-data/xbert_outputs/proc_data/C.trn.pifa-neural-s0.npz',
                'tst_label_path': './data/intermediary-data/xbert_outputs/proc_data/C.tst.pifa-neural-s0.npz',
                'output_dir': './data/intermediary-data/xbert_outputs/pifa-neural-s0/matcher',
                'config_name': '',
                'cache_dir': '',
                'do_train': False,
                'do_eval': True,
                'hidden_dropout_prob': 0.1,
                'per_device_train_batch_size': 8,
                'per_device_eval_batch_size': 16,
                'gradient_accumulation_steps': 1,
                'learning_rate': 5e-05,
                'weight_decay': 0.0,
                'adam_epsilon': 1e-08,
                'max_grad_norm': 1.0,
                'num_train_epochs': 5.0,
                'max_steps': -1,
                'warmup_steps': 0,
                'logging_steps': 100,
                'loss_func': 'l2-hinge',
                'margin': 1.0,
                'only_topk': 10,
                'no_cuda': False,
                'overwrite_output_dir': False,
                'overwrite_cache': False,
                'seed': 42,
                'fp16': False,
                'fp16_opt_level': 'O1',
                'local_rank': -1,
                'n_gpu': 1,
                'device': "cpu", # device(type='cuda', index=0),
                'eval_batch_size': 16
            }),
            X1,
            C_tst,
            topk=TOPk,
            get_hidden=True
        )

        np.save(f_X2, tst_embeddings)
        X2 = HierarchicalMLModel.load_feature_matrix()
    X = smat.hstack([
        sk_normalize(X1, axis=1),
        sk_normalize(X2, axis=1)]
    ).tocsr()
    return X

def inference_factory(icd_9_codes:t.List[str],
                      ranker_folder,
                      transform="noop",
                      beam_size=10):

    model = LinearModel.load(ranker_folder)[-1]

    def inference(discharge_summary: str):
        # TODO: embedding step
        # Xt = embedding(discharge_summary)
        Xt = None
        # TODO: add the _pred.npz generation code here
        # tempfile.open("C_pred.npz") as f:
        #     f.write(...)
        # csr_codes = "C_pred.npz"
        csr_codes = None
        Y_pred = model.predict(
            Xt,
            topk=TOPk,
            csr_codes=csr_codes,
            beam_size=beam_size,
            cond_prob=PostProcessor.get(transform)
        )
        assert len(icd_9_codes) == len(Y_pred)
        return dict(zip(icd_9_codes, Y_pred))

    return inference


if __name__ == "__main__":
    icd9_codes = open('data/intermediary-data/xbert_inputs/label_map.txt').read().strip().split()
    Y_pred = inference_factory(
        icd9_codes,
        './data/intermediary-data/xbert_outputs/pifa-tfidf-s0/ranker'
    )('Admission Date: [**2191-7-4**]        Discharge Date: [**2191-7-8**]\n\nDate of Birth:  [**2130-7-4**]        Sex:  M\n\nService:  CSU\n\n\nHISTORY OF PRESENT ILLNESS:  This is a 60-year-old gentleman\nwith a history of elevated cholesterol and hypertension and\npositive family history and HIV positive status with history\nof angina and known coronary artery disease with prior\ncatheterization in [**2186**].  He has been medically managed since\nthen, but he has now recently begun to have some chest pain\nwith meals, which is relieved by Tums and rest.  On [**2191-6-7**],\nhe had a positive exercise tolerance test with left\nventricular ejection fraction of 48 percent.  His ejection\nfraction in [**2186**] on catheterization was 50 percent.  He\nunderwent cardiac catheterization on [**2191-6-28**], which showed\nnormal left main diffuse disease in his LAD with serial 60 to\n80 percent lesions.  His circumflex artery had a high OM\nbranch with a 90 percent stenosis and 80 percent mid\nstenosis.  His RCA had a 100 percent proximal occlusion.\nEjection fraction was 58 percent with no mitral\nregurgitation.  He was referred to Dr. [**Last Name (STitle) **] for a coronary\nartery bypass grafting.\n\nPAST MEDICAL HISTORY:  Coronary artery disease, status post\ncatheterization in [**2186**].\n\nHypertension.\n\nHypercholesterolemia.\n\nHIV positive.\n\nStatus post cerebral aneurysm clipping, [**2171**].\n\nGERD.\n\nStatus post right total hip replacement times 4, last one in\n[**2182**] secondary to Staphylococcus infection.\n\nStatus post appendectomy.\n\n\nMEDICATIONS ON ADMISSION:\n1. Atenolol 50 mg p.o. q.d.\n2. Univasc 15 mg p.o. b.i.d.\n3. Norvasc 5 mg p.o. q.d.\n4. Lipitor 20 mg p.o. q.d.\n5. Aspirin 81 mg p.o. q.d.\n6. Prevacid 30 mg p.o. q.d.\n7. Zyrtec 10 mg p.o. q.d.\n8. Rhinocort nasal spray as needed.\n9. Celebrex 200 mg p.o. q.d.\n10.      Ziagen 300 mg p.o. b.i.d.\n11.      Viread 300 mg p.o. q.d.\n12.      Epivir 300 mg p.o. q.d.\n13.      Sustiva 600 mg p.o. q.d.\n14.      AndroGel topical patch.\n15.      Folate 400 mg p.o. q.d.\n16.      Vitamin C 1000 mg p.o. q.d.\n17.      Vitamin E 400 units p.o. q.d.\n18.      Fish oil q.d.\n\n\nALLERGIES:  He had no known allergies.\n\nFAMILY HISTORY:  His family history was positive for CAD.\n\nSOCIAL HISTORY:  He quit smoking 30 years ago.  He has 1 to 2\nglasses of wine per night and lives with his partner.\n\nREVIEW OF SYSTEMS:  On examination, his review of systems is\nunremarkable.  He is in no apparent distress.  Please refer\nto his medical history above.\n\nPHYSICAL EXAMINATION:  He is 5 foot 10 inches tall with a\nweight of 157 pounds.  His pupils were equal and reactive to\nlight and accommodation.  EOMs were intact.  His oropharynx\nwas benign.  His neck was supple.  He had no lymphadenopathy\nor thyromegaly.  His carotids were 2 plus bilaterally.  His\nlungs were clear to auscultation.  His heart had normal\nsounds with S1 and S2, and no murmur, rub, or gallop.  His\nabdomen was soft and nontender without any masses or\nhepatosplenomegaly, with positive bowel sounds.  His\nextremities had no clubbing, cyanosis, or edema.  His pulses\nwere 2 plus bilaterally throughout.  His neuro exam was\nnonfocal.\n\nHe was referred to Dr. [**Last Name (STitle) **].\n\nPREOPERATIVE LABORATORY DATA:  White count 4.7, hematocrit\n37, and platelets count 123,000.  PT 14.6, PTT 25.2, and INR\n1.4.  His urinalysis was negative.  Glucose 97, BUN 26,\ncreatinine 0.8, sodium 139, potassium 3.8, chloride 106,\nbicarbonate 24 with an anion gap of 13.  ALT 21, AST 21,\nalkaline phosphatase 68, amylase 53, total bilirubin 0.3, and\nalbumin 4.2.  His vitamin B12 level was 836.  His\npreoperative chest x-ray showed no acute cardiopulmonary\nabnormality.\n\nHOSPITAL COURSE:  On [**2191-7-4**], he underwent coronary artery\nbypass grafting times 3 with LIMA to the LAD, a vein graft to\nthe OM1 and a vein graft to OM2.  He was transferred to the\ncardiothoracic ICU in stable condition on titrated\nphenylephrine drip and propofol drip.  On postoperative day\n1, he was atrial paced at a rate of 90.  Another cardiac\nindex was 3.47.  Blood pressure of 123/50 on CPAP with\npostoperative labs of hematocrit 24.8 and white count 5.5\nwith the platelet count of 80,000.  Sodium was 141, potassium\n4.2, chloride 109, CO2 28, BUN 11, creatinine 0.6 with blood\nsugar of 91.  He had breath sounds bilaterally.  His abdomen\nwas soft.  His heart was regular in rate and rhythm.  He was\nweaned to extubation and pulmonary toilet was begun.  He was\non insulin drip, neo drip, and nitroglycerin drip as well as\npropofol at that time, and also continued with his\nperioperative antibiotics.  He was extubated.  The patient\nwas transferred to the floor on the afternoon of [**2191-7-5**] on\npostoperative day 1.\n\nOn postoperative day 2, the patient had some complaints of\nnausea, which was relieved by Zofran.  He otherwise had no\ncomplaints.  He had a good pain control.  He was in sinus\nrhythm at a rate of 80 with blood pressure of 130/80.  His\nhematocrit rose slightly to 27.9 with a white count of 6.8,\nand creatinine of 0.7.  His lungs were clear bilaterally with\ndecreased breath sounds at the left base.  His sternum was\nstable.  His heart was regular in rate and rhythm with normal\nsounds.  His left endoscopic harvesting sites for\nsaphenectomy were clean, dry, and intact.  His chest tubes\nwere discontinued.  His Lopressor was increased to 25 mg p.o.\nb.i.d. as he began beta-blockade.  He was seen by physical\ntherapy and begun ambulation.  He was also seen by case\nmanagement for evaluation of VNA services when he goes home.\n\nOn postoperative day 3, the patient was doing extremely well,\nambulating, he was alert, awake, and oriented with a nonfocal\nexam.  His heart was regular in rate and rhythm with no\nmurmur.  His lung sounds were clear bilaterally.  He had\nbowel sounds.  He had no edema in his extremities.  All of\nhis incisions were clean, dry, and intact.  On the evening of\n[**2191-7-7**], he did have a little bit of serosanguinous drainage\nfrom his left pleural tube site.  Of note, also his platelet\ncount decreased to 93, his pacing wires were discontinued.\nHe did have flight of stairs and his Lasix was decreased for\ndiuresis, as he was rapidly approaching his preoperative\nweight.\n\nOn postoperative day 4, the day of discharge, [**2191-7-8**], his\nLopressor was increased to 50 mg p.o. b.i.d.  His exam was as\nfollows:  Temperature 97.2 degrees, blood pressure 138/76,\nheart rate 82 and regular, respiratory rate 18, and\nsaturating 95 percent on room air.  His weight today at\ndischarge was 71.4 kg, this is approximately half kilogram\nbelow his preoperative weight.  His heart was in regular rate\nand rhythm.  He had S1 and S2, normal heart sounds with no\nmurmur.  His lungs were clear bilaterally except for\ndecreased breath sounds at both bases.  His abdomen was soft,\nnontender, and nondistended with hypoactive bowel sounds.\nHis left leg saphenectomy IVH sites were clean, dry, and\nintact with no erythema.  He had no peripheral edema that was\ndetected.  Sternum was stable, clean, dry, and intact with no\nerythema.  He had minimal serosanguinous drainage at his left\npleural tube site.  His chest x-ray from [**2191-7-6**] showed a\nsmaller pleural effusion and question of a left small apical\npneumothorax.\n\nOn the day of discharge, his labs were as follows, white\ncount 6.8, hematocrit 29.6, and platelet count rose to 136,\nso the patient was restarted on his aspirin.  Sodium 142,\npotassium 3.9, chloride 105, CO2 28, BUN 9, creatinine 0.8,\nwith a blood sugar of 101, and magnesium 2.2.\n\nDISCHARGE MEDICATIONS:\n1. Colace 100 mg p.o. b.i.d.\n2. Percocet 5/325 1 to 2 tablets p.o. p.r.n. q. 4-6h. for\n   pain.\n3. Efavirenz 600 mg p.o. q.h.s.\n4. Lamivudine 300 mg p.o. q.d.\n5. Tenofovir 300 mg p.o. q.d.\n6. Testosterone 2.5 mg 24-hour patch 1 patch q.d.\n7. Abacavir 300 mg p.o. b.i.d.\n8. Lipitor 20 mg p.o. q.d.\n9. Vitamin C 1000 mg p.o. q.d.\n10.      Lansoprazole 30 mg p.o. q.d.\n11.      Vitamin E 400 units p.o. q.d.\n12.      Metoprolol 50 mg p.o. b.i.d.\n13.      Lasix 20 mg p.o. q.d. times 5 days.\n14.      KCl 20 mEq p.o. q.d. times 5 days.\n\n\nDISCHARGE INSTRUCTIONS:  The patient was given discharge\ninstructions to follow up with Dr. [**First Name11 (Name Pattern1) 449**] [**Initial (NamePattern1) **] [**Last Name (NamePattern4) 2392**], his\nprimary care physician, [**Last Name (NamePattern4) **] 1 to 2 weeks and to see Dr.\n[**Last Name (STitle) **] in the office for postoperative visit in\napproximately 4 weeks.\n\nDISCHARGE DIAGNOSES:  Status post coronary artery bypass\ngrafting times 3.\n\nHistory of coronary artery disease.\n\nHypertension.\n\nHypercholesterolemia.\n\nPositive human immunodeficiency virus status.\n\nStatus post cerebral aneurysm clipping, [**2171**].\n\nGastroesophageal reflux disease.\n\nStatus post right total hip replacement times 4 secondary to\nstaphylococcus infection.\n\nStatus post appendectomy.\n\n\nCONDITION ON DISCHARGE:  The patient was discharged to home\nin stable condition on [**2191-7-8**].\n\n\n\n                        [**Name6 (MD) **] [**Name8 (MD) **], M.D. [**MD Number(2) 5897**]\n\nDictated By:[**Last Name (NamePattern1) **]\nMEDQUIST36\nD:  [**2191-7-8**] 09:57:49\nT:  [**2191-7-8**] 18:23:27\nJob#:  [**Job Number 32144**]\n')
    print(Y_pred)
