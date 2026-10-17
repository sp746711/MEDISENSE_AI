"""Anatomical Body-Region & X-Ray Examination Catalogue for MediSense AI.

Defines a comprehensive, hierarchical, and extensible catalogue of anatomical
regions, subregions, X-ray examination types, and common search synonyms.
Separates anatomical catalogue availability from AI vision model support.
"""

from __future__ import annotations

from typing import Any, Optional

from app.ai.xray_router import XRAY_MODEL_REGISTRY, get_model_registry_entry

# Centralized Anatomical Catalogue
ANATOMICAL_CATALOGUE: list[dict[str, Any]] = [
    {
        "id": "head_neck",
        "name": "Head & Neck",
        "category_badge": "Craniofacial",
        "description": "Skull, facial bones, orbits, paranasal sinuses, and cervical soft tissues.",
        "aliases": [
            "head", "skull", "cranium", "face", "facial", "brain", "jaw",
            "mandible", "sinus", "sinuses", "orbits", "neck", "throat",
        ],
        "subregions": [
            {
                "id": "skull",
                "name": "Skull / Cranium",
                "aliases": ["head", "skull", "cranium", "calvarium", "brain case", "head x-ray"],
                "xray_types": [
                    {
                        "id": "skull_ap_lat",
                        "name": "Skull X-Ray (AP & Lateral)",
                        "canonical_region": "skull",
                        "views": "Anteroposterior (AP), Lateral",
                        "indications": "Head injury, suspected cranial fracture, bone lesions, structural evaluation",
                        "aliases": ["skull", "head", "cranium", "skull ap lat", "head injury"],
                        "default_model_support": "unsupported",
                    },
                    {
                        "id": "skull_townes",
                        "name": "Skull X-Ray (Towne's View)",
                        "canonical_region": "skull",
                        "views": "AP Axial 30-degree caudad (Towne's)",
                        "indications": "Occipital bone evaluation, foramen magnum, petrous ridges",
                        "aliases": ["townes", "occipital", "posterior skull", "towne view"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "facial_bones",
                "name": "Facial Bones & Mandible",
                "aliases": ["face", "facial", "cheek", "jaw", "mandible", "chin", "orbits", "zygoma"],
                "xray_types": [
                    {
                        "id": "facial_bones_waters",
                        "name": "Facial Bones X-Ray (Waters & PA)",
                        "canonical_region": "facial bones",
                        "views": "Parietoacanthial (Waters), PA Axial (Caldwell)",
                        "indications": "Midface trauma, orbital blowout fracture, zygomatic arch fracture",
                        "aliases": ["facial bones", "face", "zygoma", "orbits", "cheekbone", "waters view"],
                        "default_model_support": "unsupported",
                    },
                    {
                        "id": "mandible_ap_obl",
                        "name": "Mandible / Jaw X-Ray (AP & Oblique)",
                        "canonical_region": "mandible",
                        "views": "PA Mandible, Bilateral Axiolateral Oblique",
                        "indications": "Mandibular trauma, jaw pain, suspected dental/angle fracture",
                        "aliases": ["jaw", "mandible", "chin", "lower jaw", "tmj", "jaw bone"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "sinuses",
                "name": "Paranasal Sinuses",
                "aliases": ["sinus", "sinuses", "nasal", "congestion", "maxillary", "frontal sinus"],
                "xray_types": [
                    {
                        "id": "sinuses_waters_caldwell",
                        "name": "Paranasal Sinuses X-Ray (Waters & Caldwell)",
                        "canonical_region": "sinuses",
                        "views": "Waters (occipitomental), Caldwell (occipitofrontal), Lateral",
                        "indications": "Acute/chronic sinusitis, air-fluid levels, mucosal thickening, polyps",
                        "aliases": ["sinuses", "sinus", "sinusitis", "maxillary sinus", "sinus headache"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "neck_soft_tissue",
                "name": "Soft Tissue Neck",
                "aliases": ["neck", "throat", "airway", "adenoids", "epiglottis"],
                "xray_types": [
                    {
                        "id": "neck_lateral_soft_tissue",
                        "name": "Soft Tissue Neck X-Ray (Lateral)",
                        "canonical_region": "neck",
                        "views": "Lateral soft-tissue extension",
                        "indications": "Upper airway patency, retropharyngeal space, croup, foreign body",
                        "aliases": ["neck soft tissue", "airway", "adenoid", "throat x-ray"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "chest_thorax",
        "name": "Chest & Thorax",
        "category_badge": "Cardiopulmonary",
        "description": "Lungs, mediastinum, cardiothoracic contour, bony ribs, and sternum.",
        "aliases": [
            "chest", "thorax", "lungs", "pulmonary", "cxr", "breathing",
            "heart", "ribs", "rib", "sternum", "breastbone", "clavicle",
        ],
        "subregions": [
            {
                "id": "chest_lungs",
                "name": "Chest (Lungs & Heart)",
                "aliases": ["chest", "lungs", "cxr", "pulmonary", "thorax", "cardiothoracic"],
                "xray_types": [
                    {
                        "id": "chest_pa_lat",
                        "name": "Chest X-Ray (PA & Lateral)",
                        "canonical_region": "chest",
                        "views": "Posteroanterior (PA), Lateral",
                        "indications": "Pneumonia, consolidation, effusion, pneumothorax, cardiomegaly, cough, dyspnea",
                        "aliases": ["chest", "chest x-ray", "chest xray", "cxr", "lungs", "chest pa"],
                        "default_model_support": "supported",
                    },
                    {
                        "id": "chest_ap_portable",
                        "name": "Chest X-Ray (AP Portable / Supine)",
                        "canonical_region": "chest",
                        "views": "Anteroposterior (AP)",
                        "indications": "Bedside/acute pulmonary evaluation, endotracheal/line placement",
                        "aliases": ["chest ap", "portable chest", "supine chest"],
                        "default_model_support": "supported",
                    },
                ],
            },
            {
                "id": "ribs_cage",
                "name": "Ribs & Bony Thorax",
                "aliases": ["rib", "ribs", "ribcage", "bony thorax", "chest wall"],
                "xray_types": [
                    {
                        "id": "ribs_unilateral",
                        "name": "Ribs X-Ray (Unilateral / Bilateral)",
                        "canonical_region": "ribs",
                        "views": "AP/PA and Oblique views of affected hemithorax",
                        "indications": "Blunt chest trauma, suspected rib fracture, flail chest, localized pleuritic pain",
                        "aliases": ["ribs", "rib fracture", "rib cage", "chest wall"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "sternum_clavicle",
                "name": "Sternum & Clavicle",
                "aliases": ["sternum", "breastbone", "clavicle", "collarbone", "ac joint"],
                "xray_types": [
                    {
                        "id": "sternum_lat",
                        "name": "Sternum X-Ray (RAO & Lateral)",
                        "canonical_region": "sternum",
                        "views": "Right Anterior Oblique (RAO), Lateral Sternum",
                        "indications": "Sternal trauma, steering wheel injury, suspected sternal body fracture",
                        "aliases": ["sternum", "breastbone", "sternal fracture"],
                        "default_model_support": "unsupported",
                    },
                    {
                        "id": "clavicle_ap_axial",
                        "name": "Clavicle X-Ray (AP & Axial)",
                        "canonical_region": "clavicle",
                        "views": "AP Clavicle, AP Axial 15-30 degree cephalad",
                        "indications": "Collarbone trauma, midshaft fracture, acromioclavicular separation",
                        "aliases": ["clavicle", "collarbone", "collar bone", "broken collarbone"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "spine_vertebrae",
        "name": "Spine & Vertebral Column",
        "category_badge": "Spine",
        "description": "Cervical, thoracic, lumbar spine, and sacrococcygeal segments.",
        "aliases": [
            "spine", "back", "neck", "vertebra", "vertebrae", "spinal",
            "cervical", "thoracic", "lumbar", "sacrum", "coccyx", "tailbone",
        ],
        "subregions": [
            {
                "id": "cervical_spine",
                "name": "Cervical Spine (Neck)",
                "aliases": ["neck", "c-spine", "cervical", "cervical spine", "neck pain"],
                "xray_types": [
                    {
                        "id": "c_spine_ap_lat",
                        "name": "Cervical Spine X-Ray (AP, Lateral & Odontoid)",
                        "canonical_region": "cervical spine",
                        "views": "AP, Lateral, Open-mouth Odontoid",
                        "indications": "Cervical trauma, whiplash, radiculopathy, degenerative disc disease, stiffness",
                        "aliases": ["c-spine", "cervical spine", "neck x-ray", "neck spine", "odontoid"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "thoracic_spine",
                "name": "Thoracic Spine (Mid-Back)",
                "aliases": ["mid back", "mid-back", "t-spine", "thoracic spine", "dorsal spine"],
                "xray_types": [
                    {
                        "id": "t_spine_ap_lat",
                        "name": "Thoracic Spine X-Ray (AP & Lateral)",
                        "canonical_region": "thoracic spine",
                        "views": "Anteroposterior (AP), Lateral",
                        "indications": "Mid-back pain, osteoporotic compression fracture, kyphosis, vertebral alignment",
                        "aliases": ["t-spine", "thoracic spine", "mid back x-ray", "dorsal spine"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "lumbar_spine",
                "name": "Lumbar Spine (Lower Back)",
                "aliases": ["lower back", "low back", "l-spine", "lumbar spine", "lumbosacral", "l5 s1", "back pain"],
                "xray_types": [
                    {
                        "id": "l_spine_ap_lat",
                        "name": "Lumbar Spine X-Ray (AP, Lateral & L5-S1)",
                        "canonical_region": "lumbar spine",
                        "views": "AP, Lateral, Coned L5-S1 junction",
                        "indications": "Chronic/acute low back pain, sciatica, spondylolisthesis, vertebral compression",
                        "aliases": ["lumbar", "l-spine", "lower back x-ray", "lumbar spine", "lumbosacral spine"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "sacrum_coccyx",
                "name": "Sacrum & Coccyx",
                "aliases": ["tailbone", "coccyx", "sacrum", "sacroiliac", "si joint", "buttock pain"],
                "xray_types": [
                    {
                        "id": "sacrum_coccyx_views",
                        "name": "Sacrum & Coccyx X-Ray (AP & Lateral)",
                        "canonical_region": "sacrum",
                        "views": "AP Axial Sacrum (cephalad), AP Axial Coccyx (caudad), Lateral",
                        "indications": "Tailbone fall, coccydynia, sacral fracture, sacroiliac joint sclerosis",
                        "aliases": ["tailbone", "coccyx", "sacrum", "tailbone x-ray", "sacrococcygeal"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "upper_limb",
        "name": "Shoulder & Upper Limb (Arm)",
        "category_badge": "Upper Extremity",
        "description": "Shoulder joint, humerus, elbow, and forearm segments.",
        "aliases": [
            "arm", "upper arm", "shoulder", "elbow", "forearm", "humerus",
            "bicep", "radius", "ulna", "upper limb",
        ],
        "subregions": [
            {
                "id": "shoulder",
                "name": "Shoulder",
                "aliases": ["shoulder", "arm", "shoulder joint", "glenohumeral", "rotator cuff area"],
                "xray_types": [
                    {
                        "id": "shoulder_ap_y",
                        "name": "Shoulder X-Ray (AP & Scapular Y)",
                        "canonical_region": "shoulder",
                        "views": "AP (Internal/External rotation), Scapular Y view, Axillary",
                        "indications": "Shoulder dislocation, rotator cuff pathology, proximal humerus fracture",
                        "aliases": ["shoulder", "shoulder joint", "dislocated shoulder", "shoulder ap"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "humerus",
                "name": "Humerus (Upper Arm)",
                "aliases": ["humerus", "upper arm", "arm", "bicep bone"],
                "xray_types": [
                    {
                        "id": "humerus_ap_lat",
                        "name": "Humerus / Upper Arm X-Ray (AP & Lateral)",
                        "canonical_region": "humerus",
                        "views": "AP and Lateral including shoulder and elbow joints",
                        "indications": "Humeral shaft trauma, mid-arm deformity, pathological fracture",
                        "aliases": ["humerus", "upper arm", "arm x-ray", "humerus fracture"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "elbow",
                "name": "Elbow",
                "aliases": ["elbow", "olecranon", "cubital", "elbow joint"],
                "xray_types": [
                    {
                        "id": "elbow_ap_lat",
                        "name": "Elbow X-Ray (AP & Lateral)",
                        "canonical_region": "elbow",
                        "views": "AP, 90-degree true Lateral, Oblique",
                        "indications": "Radial head fracture, posterior fat pad sign, joint effusion, dislocation",
                        "aliases": ["elbow", "elbow joint", "tennis elbow bone", "olecranon"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "forearm",
                "name": "Forearm",
                "aliases": ["forearm", "radius", "ulna", "arm"],
                "xray_types": [
                    {
                        "id": "forearm_ap_lat",
                        "name": "Forearm X-Ray (AP & Lateral)",
                        "canonical_region": "forearm",
                        "views": "AP and Lateral including wrist and elbow joints",
                        "indications": "Both-bone fracture, Monteggia / Galeazzi fracture-dislocation",
                        "aliases": ["forearm", "radius ulna", "forearm fracture", "lower arm"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "wrist_hand",
        "name": "Wrist & Hand",
        "category_badge": "Distal Upper Extremity",
        "description": "Carpal bones, scaphoid, metacarpals, fingers, and thumb.",
        "aliases": [
            "wrist", "hand", "fingers", "thumb", "carpal", "scaphoid",
            "palm", "digits", "knuckle", "wrist pain", "hand pain",
        ],
        "subregions": [
            {
                "id": "wrist",
                "name": "Wrist",
                "aliases": ["wrist", "carpal", "carpus", "distal radius", "scaphoid", "wrist pain"],
                "xray_types": [
                    {
                        "id": "wrist_pa_lat",
                        "name": "Wrist X-Ray (PA, Lateral & Oblique)",
                        "canonical_region": "wrist",
                        "views": "PA, Lateral, Semi-pronated Oblique",
                        "indications": "Colles fracture, distal radius fracture, carpal instability, arthritis",
                        "aliases": ["wrist", "wrist x-ray", "broken wrist", "wrist pa"],
                        "default_model_support": "unsupported",
                    },
                    {
                        "id": "wrist_scaphoid",
                        "name": "Scaphoid Series X-Ray",
                        "canonical_region": "wrist",
                        "views": "PA with ulnar deviation, 4-view Scaphoid series",
                        "indications": "Anatomical snuffbox tenderness, occult scaphoid waist fracture",
                        "aliases": ["scaphoid", "snuffbox", "navicular", "scaphoid fracture"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "hand",
                "name": "Hand",
                "aliases": ["hand", "palm", "metacarpals", "hand x-ray"],
                "xray_types": [
                    {
                        "id": "hand_pa_obl",
                        "name": "Hand X-Ray (PA & Oblique)",
                        "canonical_region": "hand",
                        "views": "PA, 45-degree Oblique, Fan Lateral",
                        "indications": "Metacarpal fracture (Boxer's fracture), crush injury, rheumatoid deformities",
                        "aliases": ["hand", "palm", "boxer fracture", "metacarpal"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "fingers_thumb",
                "name": "Fingers & Thumb",
                "aliases": ["finger", "fingers", "thumb", "digits", "knuckle", "phalanges"],
                "xray_types": [
                    {
                        "id": "finger_pa_lat",
                        "name": "Finger / Thumb X-Ray (PA & Lateral)",
                        "canonical_region": "fingers",
                        "views": "PA, Lateral of isolated affected digit",
                        "indications": "Phalangeal fracture, mallet finger, interphalangeal dislocation, avulsion",
                        "aliases": ["finger", "thumb", "digits", "broken finger", "pinky"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "abdomen_pelvis",
        "name": "Abdomen & Digestive",
        "category_badge": "Abdominal",
        "description": "Plain abdominal radiography, kidney-ureter-bladder (KUB), and bowel gas pattern.",
        "aliases": [
            "abdomen", "belly", "stomach", "tummy", "abdominal", "kub",
            "gut", "bowel", "digestive", "kidneys", "bladder",
        ],
        "subregions": [
            {
                "id": "abdomen_plain",
                "name": "Abdomen (Plain Film / KUB)",
                "aliases": ["abdomen", "belly", "stomach", "tummy", "kub", "bowel"],
                "xray_types": [
                    {
                        "id": "abdomen_supine_erect",
                        "name": "Abdomen X-Ray / KUB (Supine & Erect)",
                        "canonical_region": "abdomen",
                        "views": "AP Supine, AP Erect (or Left Lateral Decubitus)",
                        "indications": "Intestinal obstruction, pneumoperitoneum (free air), nephrolithiasis, ileus",
                        "aliases": ["abdomen", "belly", "kub", "kidney stone", "bowel obstruction", "plain abdomen"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "pelvis_hip",
        "name": "Pelvis, Hip & Femur",
        "category_badge": "Pelvic & Thigh",
        "description": "Pelvic girdle, hip joint, femoral neck, and thigh shaft.",
        "aliases": [
            "pelvis", "pelvic", "hip", "femur", "thigh", "groin",
            "acetabulum", "hip joint", "broken hip", "pelvis pain",
        ],
        "subregions": [
            {
                "id": "pelvis",
                "name": "Pelvis",
                "aliases": ["pelvis", "pelvic", "pelvic ring", "iliac", "pubic"],
                "xray_types": [
                    {
                        "id": "pelvis_ap",
                        "name": "Pelvis X-Ray (AP)",
                        "canonical_region": "pelvis",
                        "views": "Anteroposterior (AP) standing or supine",
                        "indications": "Pelvic ring disruption, sacroiliac diastasis, pubic rami fractures, avulsions",
                        "aliases": ["pelvis", "pelvic x-ray", "pelvic ring"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "hip",
                "name": "Hip Joint",
                "aliases": ["hip", "femoral neck", "groin", "hip pain", "hip replacement"],
                "xray_types": [
                    {
                        "id": "hip_ap_frog",
                        "name": "Hip X-Ray (AP & Frog-Leg Lateral)",
                        "canonical_region": "hip",
                        "views": "Unilateral AP Hip, Cleaves / Frog-leg Lateral",
                        "indications": "Femoral neck fracture, subcapital fracture, osteoarthritis, avascular necrosis",
                        "aliases": ["hip", "hip joint", "broken hip", "femoral neck fracture", "hip fracture"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "femur",
                "name": "Femur (Thigh)",
                "aliases": ["femur", "thigh", "upper leg", "thigh bone"],
                "xray_types": [
                    {
                        "id": "femur_ap_lat",
                        "name": "Femur / Thigh X-Ray (AP & Lateral)",
                        "canonical_region": "femur",
                        "views": "AP and Lateral including hip and knee joints",
                        "indications": "Femoral shaft high-energy trauma, stress fracture, osteolytic lesion",
                        "aliases": ["femur", "thigh", "femur fracture", "thigh x-ray"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "lower_limb",
        "name": "Knee & Lower Leg",
        "category_badge": "Lower Extremity",
        "description": "Knee joint, patella, tibia, and fibula segments.",
        "aliases": [
            "knee", "leg", "lower leg", "patella", "kneecap", "tibia",
            "fibula", "shin", "calf", "knee pain", "leg pain",
        ],
        "subregions": [
            {
                "id": "knee",
                "name": "Knee",
                "aliases": ["knee", "patella", "kneecap", "knee joint", "tibiofemoral"],
                "xray_types": [
                    {
                        "id": "knee_ap_lat",
                        "name": "Knee X-Ray (AP, Lateral & Skyline)",
                        "canonical_region": "knee",
                        "views": "Weight-bearing AP, 30-degree Lateral, Sunrise/Merchant patellar view",
                        "indications": "Tibial plateau fracture, patellar fracture, osteoarthritis, joint effusion",
                        "aliases": ["knee", "kneecap", "patella", "knee x-ray", "knee pain"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "lower_leg",
                "name": "Lower Leg (Tibia & Fibula)",
                "aliases": ["lower leg", "tibia", "fibula", "shin", "calf", "leg"],
                "xray_types": [
                    {
                        "id": "lower_leg_ap_lat",
                        "name": "Lower Leg X-Ray (Tibia & Fibula AP/Lateral)",
                        "canonical_region": "lower leg",
                        "views": "AP and Lateral including knee and ankle joints",
                        "indications": "Tibial shaft fracture, fibular fracture, compartment syndrome trauma evaluation",
                        "aliases": ["lower leg", "tibia", "fibula", "shin", "broken shin", "calf x-ray"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
    {
        "id": "ankle_foot",
        "name": "Ankle & Foot",
        "category_badge": "Distal Lower Extremity",
        "description": "Ankle mortise, tarsals, metatarsals, heel bone, and toes.",
        "aliases": [
            "ankle", "foot", "feet", "heel", "calcaneus", "toes",
            "instep", "malleolus", "ankle sprain", "foot pain",
        ],
        "subregions": [
            {
                "id": "ankle",
                "name": "Ankle",
                "aliases": ["ankle", "malleolus", "ankle joint", "ankle sprain"],
                "xray_types": [
                    {
                        "id": "ankle_ap_mortise",
                        "name": "Ankle X-Ray (AP, Mortise & Lateral)",
                        "canonical_region": "ankle",
                        "views": "AP, 15-20 degree internal oblique (Mortise), Lateral",
                        "indications": "Ottawa ankle rules, bimalleolar/trimalleolar fracture, ligamentous tear",
                        "aliases": ["ankle", "ankle x-ray", "sprained ankle", "broken ankle", "malleolus"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "foot",
                "name": "Foot",
                "aliases": ["foot", "feet", "metatarsal", "tarsal", "midfoot", "lisfranc"],
                "xray_types": [
                    {
                        "id": "foot_ap_obl",
                        "name": "Foot X-Ray (AP, Oblique & Lateral)",
                        "canonical_region": "foot",
                        "views": "AP with 10-degree cephalic tube tilt, 30-degree Medial Oblique, Lateral",
                        "indications": "Metatarsal stress fracture, 5th metatarsal Jones fracture, Lisfranc injury",
                        "aliases": ["foot", "feet", "broken foot", "jones fracture", "foot x-ray"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
            {
                "id": "calcaneus_toes",
                "name": "Calcaneus (Heel) & Toes",
                "aliases": ["heel", "calcaneus", "toes", "toe", "big toe", "heel spur", "heel pain"],
                "xray_types": [
                    {
                        "id": "calcaneus_axial",
                        "name": "Calcaneus / Heel X-Ray (Axial & Lateral)",
                        "canonical_region": "calcaneus",
                        "views": "Plantodorsal Axial 40-degree, Lateral Calcaneus",
                        "indications": "Fall from height, Lover's fracture, calcaneal spur, subtalar joint evaluation",
                        "aliases": ["heel", "calcaneus", "heel bone", "heel pain", "calcaneal fracture"],
                        "default_model_support": "unsupported",
                    },
                    {
                        "id": "toes_ap_obl",
                        "name": "Toes X-Ray (AP & Oblique)",
                        "canonical_region": "toes",
                        "views": "AP, Medial Oblique of affected toes",
                        "indications": "Direct stubbing injury, phalangeal fracture, dislocation",
                        "aliases": ["toes", "toe", "broken toe", "big toe", "little toe"],
                        "default_model_support": "unsupported",
                    },
                ],
            },
        ],
    },
]


def get_xray_catalogue_with_model_status() -> list[dict[str, Any]]:
    """Enrich the anatomical catalogue with real-time model capability disclosures."""
    enriched_catalogue = []

    for main_region in ANATOMICAL_CATALOGUE:
        region_copy = dict(main_region)
        enriched_subregions = []

        for subreg in main_region.get("subregions", []):
            subreg_copy = dict(subreg)
            enriched_xray_types = []

            for xtype in subreg.get("xray_types", []):
                xtype_copy = dict(xtype)
                canon_reg = xtype.get("canonical_region", "unknown")
                reg_entry = get_model_registry_entry(canon_reg)

                if reg_entry and reg_entry.is_supported:
                    is_avail, avail_msg = reg_entry.check_availability()
                    status_label = "supported"
                    badge_label = "Supported (ResNet-18)" if is_avail else "Supported (Model Configured)"
                    notice = (
                        "Automated computer-vision screening pipeline is configured for chest radiography (ResNet-18). "
                        "If model weights are loaded on the server, automated screening will execute."
                    )
                else:
                    is_avail = False
                    status_label = "unsupported"
                    badge_label = "Unsupported (No CV Model)"
                    reg_display = reg_entry.display_name if reg_entry else canon_reg.title()
                    notice = (
                        f"Automated AI vision inference is currently unavailable for {reg_display} radiography. "
                        "Your image and any accompanying radiology reports will be safely preserved and documented in your assessment feedback."
                    )

                xtype_copy["model_support"] = {
                    "status": status_label,
                    "is_supported": bool(reg_entry and reg_entry.is_supported),
                    "is_available": is_avail,
                    "badge": badge_label,
                    "notice": notice,
                    "model_version": reg_entry.model_version if reg_entry else None,
                    "model_architecture": reg_entry.model_architecture if reg_entry else None,
                }
                enriched_xray_types.append(xtype_copy)

            subreg_copy["xray_types"] = enriched_xray_types
            enriched_subregions.append(subreg_copy)

        region_copy["subregions"] = enriched_subregions
        enriched_catalogue.append(region_copy)

    return enriched_catalogue


# Clean, simplified searchable body-regions catalogue for streamlined user search
SEARCHABLE_BODY_REGIONS: list[dict[str, Any]] = [
    {
        "id": "head",
        "name": "Head",
        "canonical_region": "skull",
        "aliases": ["head", "skull", "cranium", "brain", "face", "facial", "sinus", "jaw", "mandible", "calvarium"],
    },
    {
        "id": "skull",
        "name": "Skull",
        "canonical_region": "skull",
        "aliases": ["skull", "head", "cranium", "calvarium", "braincase"],
    },
    {
        "id": "chest",
        "name": "Chest",
        "canonical_region": "chest",
        "aliases": ["chest", "lungs", "cxr", "thorax", "pulmonary", "ribs", "sternum", "breathing", "heart"],
    },
    {
        "id": "spine",
        "name": "Spine",
        "canonical_region": "spine",
        "aliases": [
            "spine", "back", "neck", "cervical", "c-spine", "thoracic",
            "t-spine", "lumbar", "l-spine", "low back", "lower back",
            "sacrum", "tailbone", "coccyx"
        ],
    },
    {
        "id": "arm",
        "name": "Arm",
        "canonical_region": "humerus",
        "aliases": ["arm", "upper arm", "humerus", "bicep"],
    },
    {
        "id": "shoulder",
        "name": "Shoulder",
        "canonical_region": "shoulder",
        "aliases": ["shoulder", "clavicle", "collarbone", "scapula", "rotator cuff"],
    },
    {
        "id": "elbow",
        "name": "Elbow",
        "canonical_region": "elbow",
        "aliases": ["elbow", "olecranon", "cubital"],
    },
    {
        "id": "forearm",
        "name": "Forearm",
        "canonical_region": "forearm",
        "aliases": ["forearm", "radius", "ulna", "arm"],
    },
    {
        "id": "wrist",
        "name": "Wrist",
        "canonical_region": "wrist",
        "aliases": ["wrist", "carpal", "scaphoid", "colles"],
    },
    {
        "id": "hand",
        "name": "Hand",
        "canonical_region": "hand",
        "aliases": ["hand", "palm", "metacarpal", "boxer"],
    },
    {
        "id": "fingers",
        "name": "Fingers / Thumb",
        "canonical_region": "fingers",
        "aliases": ["fingers", "finger", "thumb", "digits", "pinky", "phalanges"],
    },
    {
        "id": "abdomen",
        "name": "Abdomen",
        "canonical_region": "abdomen",
        "aliases": ["abdomen", "belly", "stomach", "tummy", "kub", "bowel", "gut"],
    },
    {
        "id": "pelvis",
        "name": "Pelvis",
        "canonical_region": "pelvis",
        "aliases": ["pelvis", "pelvic", "iliac", "pubic", "pelvic ring"],
    },
    {
        "id": "hip",
        "name": "Hip",
        "canonical_region": "hip",
        "aliases": ["hip", "groin", "femoral neck", "acetabulum"],
    },
    {
        "id": "femur",
        "name": "Femur / Thigh",
        "canonical_region": "femur",
        "aliases": ["femur", "thigh", "upper leg"],
    },
    {
        "id": "knee",
        "name": "Knee",
        "canonical_region": "knee",
        "aliases": ["knee", "patella", "kneecap"],
    },
    {
        "id": "lower_leg",
        "name": "Lower Leg",
        "canonical_region": "lower leg",
        "aliases": ["lower leg", "shin", "tibia", "fibula", "leg"],
    },
    {
        "id": "ankle",
        "name": "Ankle",
        "canonical_region": "ankle",
        "aliases": ["ankle", "malleolus", "mortise"],
    },
    {
        "id": "foot",
        "name": "Foot",
        "canonical_region": "foot",
        "aliases": ["foot", "feet", "metatarsal", "calcaneus", "heel", "toes"],
    },
    {
        "id": "sinuses",
        "name": "Sinuses",
        "canonical_region": "sinuses",
        "aliases": ["sinus", "sinuses", "paranasal", "nasal"],
    },
    {
        "id": "facial_bones",
        "name": "Facial Bones",
        "canonical_region": "facial bones",
        "aliases": ["face", "facial", "cheekbone", "jaw", "mandible", "chin", "orbits"],
    },
    {
        "id": "ribs",
        "name": "Ribs",
        "canonical_region": "ribs",
        "aliases": ["ribs", "rib", "ribcage", "chest wall"],
    },
]


def get_backend_body_regions() -> list[dict[str, Any]]:
    """Return backend-driven list of distinct body regions with live model status and search aliases."""
    results = []
    for item in SEARCHABLE_BODY_REGIONS:
        canon = item["canonical_region"]
        entry = get_model_registry_entry(canon)
        is_sup = bool(entry and entry.is_supported)
        is_avail = False
        if entry and is_sup:
            is_avail, _ = entry.check_availability()
            notice = "Automated AI vision screening pipeline is configured for chest radiography (ResNet-18)."
            model_status = "supported"
        else:
            notice = f"Automated AI vision inference is currently unavailable for {item['name']} radiography. Your image and reports will be safely preserved without fabricated diagnoses."
            model_status = "unsupported"

        results.append({
            "id": item["id"],
            "name": item["name"],
            "canonical_region": canon,
            "aliases": item["aliases"],
            "is_supported": is_sup,
            "is_available": is_avail,
            "model_status": model_status,
            "model_notice": notice,
        })
    return results

