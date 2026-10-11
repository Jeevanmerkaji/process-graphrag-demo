# Virtual Validation (CAE): process overview

*Synthetic example document for demonstration purposes.*

| Step | Responsible | Input | Output | System | Pain points |
|---|---|---|---|---|---|
| Define load cases | Simulation Engineer | Requirements Specification, CAD Model | Load Case Definition | Requirements management tool | Load cases are re-defined from scratch for each project; existing definitions are hard to find |
| Prepare simulation model | Simulation Engineer | CAD Model | Simulation Model | CAE pre-processor | Meshing takes about 40% of the total simulation effort |
| Run simulation | Simulation Engineer | Simulation Model, Load Case Definition | Simulation Results | HPC cluster | Long solver run times limit the number of design variants |
| Evaluate and report | Simulation Engineer | Simulation Results | Simulation Report | CAE post-processor | Reports are written manually; comparison with earlier variants is tedious |
| Correlate with test | Simulation Engineer, Test Engineer | Simulation Report, Test Report | Correlation Report | - | Test and simulation data are stored in different systems |

**Notes:** the Simulation Report is used as evidence in the Design FMEA and is presented at the quality gate review.
