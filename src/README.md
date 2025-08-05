Code structure:

```
zero_order_optim/
├── __init__.py
├── base.py
├── perturbations.py
├── optimizers/
│   ├── __init__.py
│   ├── zo_sgd.py
│   ├── zo_adam.py
│   ├── zo_conservative.py
│   ├── zo_sign_sgd.py
│   ├── forward_grad.py
│   ├── jaguar_sign_sgd.py
│   ├── jaguar_muon.py
│   └── zo_muon.py
└── utils.py
```