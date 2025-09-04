# Portfolio Optimization Demo (Streamlit)

## C 11b 6rai

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ch a y app

```bash
streamlit run app.py
```

## D f 6 lieu dau vao
- File CSV `df_return` dang n x m, moi cot la loi nhuan cua 1 co phieu, moi dong la 1 ngay.
- Neu khong upload, app se sinh du lieu gia lap.

## Su dung
- Chon short-list co phieu o sidebar.
- Chon ham muc tieu: Markowitz (mean-variance) hoac Sharpe.
- Chon thuat toan: GD, mini-batch GD, SGD, Newton, Nesterov, Adam.
- Nhan nut `RUN_STEP` de thuc hien 1 buoc toi uu, hien thi `weights` va `objective`.
- Neu short-list co 2 co phieu, app ve bieu do contour gia tri ham muc tieu.

## Luu y
- Bai toan toi uu khong rang buoc, trong so co the am (ban khong) hoac > 1 (don bay).

---

# So sanh thuat toan (app2.py)

Ch a y app so s anh:

```bash
streamlit run app2.py
```

- Chon cac thuat toan muon so sanh (GD, mini-batch, SGD, Newton, Nesterov, Adam).
- Chon ham muc tieu (Markowitz/Sharpe) va tham so (lambda, r_f), hyperparams (learning rate, batch size, so iterations, do tre cap nhat).
- Nhan `RUN` de chay song song, cap nhat theo thoi gian thuc.
- Bieu do 1: Contour + quy dao (k=2).
- Bieu do 2: Objective theo iteration.
- Bieu do 3: Objective theo thoi gian (giay).
