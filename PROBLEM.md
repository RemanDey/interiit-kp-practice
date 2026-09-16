3: The Bin at the End of the Belt (450pts)
Background & Context
Inside the high-throughput sortation hubs of Apex Logistics, robotic arms with pneumatic suction
cups and two-finger parallel grippers sit above conveyor terminals. The bins arriving at these
stations are physical chaos. They do not contain neat rows of pristine boxes; they are filled with
tangled charging cables, crushed parcels wrapped in highly reflective cellophane, semi-transparent
plastic water bottles, and odd-shaped retail goods tossed on top of one another. 
When a human picker looks into the bin and hears "grab the semi-crushed blue carton wedged
beneath the foam roller," they effortlessly parse occlusions, deduce depth despite specular glares,
infer which object must be moved first, and position their fingers along stable contact surfaces. The
facility's automated vision system, however, fails constantly. Standard vision models identify the 2D
bounding boxes, but when the robotic arm plunges into the bin, the suction fails against curved
surfaces, or the gripper collides with an unmapped cardboard edge, triggering an emergency stop.
The Challenge
Apex wants to move away from rigid, pre-scanned 3D CAD libraries and brittle depth thresholding.
They need an end-to-end spatial understanding system. Fed only RGB image feeds, and free-form
linguistic instructions, the system must navigate visual ambiguity, understand the 3D space, and
output reliable, serial navigation instructions.
Whether you rely on continuous implicit representations, open-vocabulary feature grounding,
geometric primitives, or something entirely unconventional is your architectural choice.
Key Deliverables
● Working spatial grounding and grasp proposal pipeline with verification scripts on cluttered
scenes.
● Technical whitepaper detailing your coordinate transformations, occlusion reasoning, and
failure-mode analysis.