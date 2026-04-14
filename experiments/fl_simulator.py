from __future__ import annotations

# Định nghĩa class FLSimulator chuẩn để import từ ngoài
class FLSimulator:

    SCALE_FACTOR = 100000000.0

    def __init__(
        self,
        model_fn:       Callable[[], nn.Module],
        loss_fn:        nn.Module,
        kem_backend:    str = "DH",  # config file
        sig_backend:    str = "classic",  # config file
        secagg_n:       int = 10,  # config file
        device:         Optional[str] = None,  # config file
        n_local_epochs: int = 1,  # config file
        lr:             float = 0.01,  # config file
    ) -> None:
        self.model_fn       = model_fn
        self.loss_fn        = loss_fn
        self.kem_backend    = kem_backend
        self.sig_backend    = sig_backend
        self.secagg_n       = secagg_n
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_local_epochs = n_local_epochs
        self.lr             = lr
        self.global_model   = model_fn().to(self.device)

    # def train_round(
    #     self,
    #     train_loaders: List[DataLoader],
    #     dropout_rate:  float = 0.0,  # config file
    # ) -> None:
    #     global_flat = _get_flat_params(self.global_model)
    #     rng = np.random.default_rng()
    #     alive = [
    #         i for i in range(len(train_loaders))
    #         if rng.random() > dropout_rate
    #     ]

    #     client_updates: List[torch.Tensor] = []
    #     for idx in alive:
    #         local_model = deepcopy(self.global_model)
    #         opt = torch.optim.SGD(local_model.parameters(), lr=self.lr,
    #                               momentum=0.9, weight_decay=1e-4)
    #         local_model.train()
    #         for _ in range(self.n_local_epochs):
    #             for X, y in train_loaders[idx]:
    #                 X, y = X.to(self.device), y.to(self.device)
    #                 opt.zero_grad()
    #                 self.loss_fn(local_model(X), y).backward()
    #                 opt.step()
    #         client_updates.append(_get_flat_params(local_model))

    #     if client_updates:
    #         new_flat = _fed_avg(global_flat, client_updates)
    #         _set_flat_params(self.global_model, new_flat)
    def train_round(
        self,
        train_loaders: List[DataLoader],
        dropout_rate: float = 0.0,
    ) -> None:
        n_clients = len(train_loaders)
        sids = [f"C{i:04d}" for i in range(n_clients)]
        # rng = np.random.default_rng()


        # 1. Giả lập rớt mạng: Sử dụng np.random.rand() để ăn theo Global Seed
        alive_indices = [i for i in range(n_clients) if np.random.rand() > dropout_rate]
        dropout_indices = [i for i in range(n_clients) if i not in alive_indices]

        survivor_sids = [sids[i] for i in alive_indices]
        dropout_sids = [sids[i] for i in dropout_indices]

        # # 1. Giả lập rớt mạng: chia thành danh sách sống (survivors) và rớt mạng (dropouts)
        # alive_indices = [i for i in range(n_clients) if rng.random() > dropout_rate]
        # dropout_indices = [i for i in range(n_clients) if i not in alive_indices]

        # survivor_sids = [sids[i] for i in alive_indices]
        # dropout_sids = [sids[i] for i in dropout_indices]

        # 2. Lấy kích thước mô hình hiện tại
        global_flat = _get_flat_params(self.global_model)
        grad_shape = (global_flat.numel(),)

        # =====================================================================
        # BƯỚC A: THIẾT LẬP MẬT MÃ CHO TẤT CẢ CLIENTS (Round 0 & 1)
        # =====================================================================
        use_mlkem = self.kem_backend != "DH"
        
        if use_mlkem:
            from secagg.crypto_mlkem import SecAggregatorMLKEM
            crypto_clients = {sid: SecAggregatorMLKEM(shape=grad_shape, security_level=self.kem_backend) for sid in sids}
            peer_eks = {sid: crypto_clients[sid].public_key for sid in sids}
            all_cts = {}
            for sid in sids:
                all_cts[sid] = crypto_clients[sid].generate_ciphertexts(peer_eks, sid)
            
            # Trao đổi ciphertexts giữa các client
            for i, sid_v in enumerate(sids):
                incoming = {
                    sid_u: all_cts[sid_u][sid_v]
                    for sid_u in sids[:i]
                    if sid_v in all_cts.get(sid_u, {})
                }
                if incoming:
                    crypto_clients[sid_v].receive_ciphertexts(incoming)
        else:
            from secagg.crypto import SecAggregator
            crypto_clients = {sid: SecAggregator(shape=grad_shape) for sid in sids}
            all_pks = {sid: crypto_clients[sid].public_key for sid in sids}

        # =====================================================================
        # BƯỚC B: HUẤN LUYỆN AI CỤC BỘ & TẠO MẶT NẠ NHIỄU
        # =====================================================================
        masked_updates = []

        # Lưu ý: Chỉ những client "sống" mới có thể hoàn thành việc train và nộp bài
        for idx in alive_indices:
            sid = sids[idx]
            local_model = deepcopy(self.global_model)
            opt = torch.optim.SGD(local_model.parameters(), lr=self.lr,
                                  momentum=0.9, weight_decay=1e-4)
            local_model.train()
            
            # Client tự học trên tập dữ liệu ảnh của mình
            for _ in range(self.n_local_epochs):
                for X, y in train_loaders[idx]:
                    X, y = X.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    self.loss_fn(local_model(X), y).backward()
                    opt.step()
            
            # Trích xuất trọng số thật đã học thành mảng numpy float64
            local_weights_np = _get_flat_params(local_model).cpu().numpy().astype(np.float64)
            
            # Đưa bộ trọng số thật này vào engine bảo mật của client
            crypto_clients[sid].set_weights(local_weights_np)
            
            # Client che giấu dữ liệu bằng mặt nạ nhiễu (Dữ liệu trả về hoàn toàn không thể đọc được)
            if use_mlkem:
                masked_data = crypto_clients[sid].prepare_masked_gradient()
            else:
                masked_data = crypto_clients[sid].prepare_masked_gradient(all_pks, sid)
                
            masked_updates.append(masked_data)

            # print(f"[{sid}] 5 trọng số THẬT: {local_weights_np[:5]}")
            # print(f"[{sid}] 5 trọng số NHIỄU: {masked_data[:5]}\n")

        # Nếu xui xẻo tất cả đều rớt mạng thì bỏ qua round này
        if not masked_updates:
            return

        # =====================================================================
        # BƯỚC C: SERVER GỠ MẶT NẠ VÀ TỔNG HỢP (FEDAVG)
        # =====================================================================
        
        # 1. Server nhận dữ liệu và cộng dồn tất cả lại (Lúc này kết quả vẫn là một đống rác vô nghĩa)
        server_aggregate = np.sum(masked_updates, axis=0)

        # 2. Server xin các "mảnh khóa gỡ rối" từ những người sống sót
        for sid in survivor_sids:
            
            # Gỡ mặt nạ riêng tư cá nhân
            priv_correction = crypto_clients[sid].private_mask()
            server_aggregate += priv_correction
            
            # Gỡ mặt nạ cặp với những client đã rớt mạng
            if dropout_sids:
                pair_correction = crypto_clients[sid].reveal_pairwise_masks(dropout_sids)
                server_aggregate += pair_correction

        # 3. Tính trung bình FedAvg
        # Kỳ diệu thay, sau Bước 2, tất cả nhiễu đã tự triệt tiêu lẫn nhau.
        # server_aggregate hiện tại chính xác là tổng của tất cả TRỌNG SỐ THẬT ban đầu.
        real_avg_weights_int = server_aggregate / len(survivor_sids)

        real_avg_weights = (real_avg_weights_int / self.SCALE_FACTOR).astype(np.float64)


        # print(f"[SERVER] Trung bình FedAvg THẬT (mong đ ợi): {(crypto_clients[sids[0]]._weights[:5] + crypto_clients[sids[1]]._weights[:5]) / 2}")
        # print(f"[SERVER] Trung bình Server GIẢI MÃ ĐƯỢC: {real_avg_weights[:5]}")
        # print("--------------------------------------------------")

        # 4. Đóng gói lại thành Tensor và cập nhật vào mô hình AI toàn cục của Server
        new_flat = torch.tensor(real_avg_weights, dtype=torch.float64).to(self.device)
        _set_flat_params(self.global_model, new_flat)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.global_model.eval()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.global_model(X)
                total_loss += self.loss_fn(logits, y).item() * len(y)
                correct    += (logits.argmax(1) == y).sum().item()
                n          += len(y)
        return total_loss / n, correct / n

    def secagg_timing(self, grad_shape: Tuple[int, ...], n_repeat: int = 3) -> PhaseTimer:
        return run_secagg_timing(
            n_clients   = self.secagg_n,
            grad_shape  = grad_shape,
            kem_backend = self.kem_backend,
            sig_backend = self.sig_backend,
            n_repeat    = n_repeat,
        )

    def reset(self) -> None:
        self.global_model = self.model_fn().to(self.device)

import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader




@dataclass
class PhaseTimer:
    advertise_keys: float = 0.0   # Round 0 — keygen + sign public key broadcast
    share_keys:     float = 0.0   # Round 1 — KEM encap/decap + ciphertext deliver
    verify_sigs:    float = 0.0   # Round 1 — verify all peer pk signatures (n²)
    masked_input:   float = 0.0   # Round 2 — compute + upload masked gradient
    unmasking:      float = 0.0   # Round 3 — reveal pairwise masks for dropouts
    total:          float = 0.0   # sum of all phases




def _get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.clone().flatten() for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset: offset + numel].view(p.shape))
        offset += numel


def _fed_avg(global_flat: torch.Tensor, client_updates: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(client_updates).mean(dim=0)




def run_secagg_timing(
    n_clients:   int,
    grad_shape:  Tuple[int, ...],
    kem_backend: str = "DH",
    sig_backend: str = "classic",
    n_repeat:    int = 3,
    dropout_rate: float = 0.2,
) -> PhaseTimer:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    use_mlkem = kem_backend != "DH"
    sids = [f"C{i:04d}" for i in range(n_clients)]
    n_drop        = max(0, min(n_clients - 1, int(round(n_clients * dropout_rate))))
    survivor_sids = sids[: n_clients - n_drop]
    dropout_sids  = sids[n_clients - n_drop :]

    from secagg.sig_pq import make_signer
    signer = make_signer(sig_backend)

    acc = PhaseTimer()

    for _ in range(n_repeat):

        # Round 0: keygen + sign public key
        t = time.perf_counter()
        if use_mlkem:
            from secagg.crypto_mlkem import SecAggregatorMLKEM
            clients = {
                sid: SecAggregatorMLKEM(shape=grad_shape, security_level=kem_backend)
                for sid in sids
            }
        else:
            from secagg.crypto import SecAggregator
            clients = {sid: SecAggregator(shape=grad_shape) for sid in sids}

        # Each client signs its public key (Round 0 advertise)
        sig_info: Dict[str, tuple] = {}
        for sid in sids:
            sig_pk, sig_sk = signer.keygen()
            raw_pk = clients[sid].public_key
            msg = (raw_pk if isinstance(raw_pk, (bytes, bytearray))
                   else hashlib.sha256(str(raw_pk).encode()).digest())
            sig = signer.sign(sig_sk, msg)
            sig_info[sid] = (sig_pk, sig, msg)
        acc.advertise_keys += time.perf_counter() - t

        # Round 1a: KEM ciphertext exchange
        t = time.perf_counter()
        if use_mlkem:
            peer_eks = {sid: clients[sid].public_key for sid in sids}
            all_cts: Dict[str, Dict] = {}
            for sid in sids:
                all_cts[sid] = clients[sid].generate_ciphertexts(peer_eks, sid)
            for i, sid_v in enumerate(sids):
                incoming = {
                    sid_u: all_cts[sid_u][sid_v]
                    for sid_u in sids[:i]
                    if sid_v in all_cts.get(sid_u, {})
                }
                if incoming:
                    clients[sid_v].receive_ciphertexts(incoming)
        # DH: non-interactive, no ciphertext step
        acc.share_keys += time.perf_counter() - t

        # Round 1b: verify all peer pk signatures
        t = time.perf_counter()
        for sid_v in survivor_sids:
            for sid_u in sids:
                if sid_u == sid_v:
                    continue
                sig_pk, sig, msg = sig_info[sid_u]
                ok = signer.verify(sig_pk, msg, sig)
                assert ok, f"SIG verify failed: {sid_u} → {sid_v}"
        acc.verify_sigs += time.perf_counter() - t

        # Round 2: masked gradient (survivors)
        t = time.perf_counter()
        if use_mlkem:
            for sid in survivor_sids:
                clients[sid].set_weights(np.ones(grad_shape, dtype=np.float64))
            for sid in survivor_sids:
                clients[sid].prepare_masked_gradient()
        else:
            all_pks = {sid: clients[sid].public_key for sid in sids}
            for sid in survivor_sids:
                clients[sid].set_weights(np.ones(grad_shape, dtype=np.float64))
                clients[sid].prepare_masked_gradient(all_pks, sid)
        acc.masked_input += time.perf_counter() - t

        # Round 3: reveal pairwise masks for dropouts
        t = time.perf_counter()
        for sid in survivor_sids:
            clients[sid].reveal_pairwise_masks(dropout_sids)
        acc.unmasking += time.perf_counter() - t

    timer = PhaseTimer(
        advertise_keys = acc.advertise_keys / n_repeat,
        share_keys     = acc.share_keys     / n_repeat,
        verify_sigs    = acc.verify_sigs    / n_repeat,
        masked_input   = acc.masked_input   / n_repeat,
        unmasking      = acc.unmasking      / n_repeat,
    )
    timer.total = (timer.advertise_keys + timer.share_keys + timer.verify_sigs
                   + timer.masked_input + timer.unmasking)
    return timer






    def __init__(
        self,
        model_fn:       Callable[[], nn.Module],
        loss_fn:        nn.Module,
        kem_backend:    str = "DH",  # config file
        sig_backend:    str = "classic",  # config file
        secagg_n:       int = 10,  # config file
        device:         Optional[str] = None,  # config file
        n_local_epochs: int = 1,  # config file
        lr:             float = 0.01,  # config file
    ) -> None:
        self.model_fn       = model_fn
        self.loss_fn        = loss_fn
        self.kem_backend    = kem_backend
        self.sig_backend    = sig_backend
        self.secagg_n       = secagg_n
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_local_epochs = n_local_epochs
        self.lr             = lr
        self.global_model   = model_fn().to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_round(
        self,
        train_loaders: List[DataLoader],
        dropout_rate:  float = 0.0,  # config file
    ) -> None:
        global_flat = _get_flat_params(self.global_model)
        rng = np.random.default_rng()

        
        alive = [
            i for i in range(len(train_loaders))
            # if rng.random() > dropout_rate
            if np.random.rand() > dropout_rate 
        ]

        client_updates: List[torch.Tensor] = []
        for idx in alive:
            local_model = deepcopy(self.global_model)
            opt = torch.optim.SGD(local_model.parameters(), lr=self.lr,
                                  momentum=0.9, weight_decay=1e-4)
            local_model.train()
            for _ in range(self.n_local_epochs):
                for X, y in train_loaders[idx]:
                    X, y = X.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    self.loss_fn(local_model(X), y).backward()
                    opt.step()
            client_updates.append(_get_flat_params(local_model))

        if client_updates:
            new_flat = _fed_avg(global_flat, client_updates)
            _set_flat_params(self.global_model, new_flat)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.global_model.eval()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.global_model(X)
                total_loss += self.loss_fn(logits, y).item() * len(y)
                correct    += (logits.argmax(1) == y).sum().item()
                n          += len(y)
        return total_loss / n, correct / n

    def secagg_timing(self, grad_shape: Tuple[int, ...], n_repeat: int = 3) -> PhaseTimer:
        return run_secagg_timing(
            n_clients   = self.secagg_n,
            grad_shape  = grad_shape,
            kem_backend = self.kem_backend,
            sig_backend = self.sig_backend,
            n_repeat    = n_repeat,
        )

    def reset(self) -> None:
        self.global_model = self.model_fn().to(self.device)




def simulate_secagg_phases(
    n_clients:     int,
    grad_shape:    Tuple[int, ...],
    crypto_backend: str = "DH",  # config file
    n_repeat:      int = 3,  # config file
) -> Dict[str, float]:
    use_mlkem = crypto_backend != "DH"
    sids = [f"C{i:04d}" for i in range(n_clients)]
    dropout_idx  = max(0, n_clients - max(1, n_clients // 5))
    survivor_sids = sids[:dropout_idx]
    dropout_sids  = sids[dropout_idx:]

    totals: Dict[str, float] = {
        "round0_keygen":     0.0,
        "round05_encaps":    0.0,
        "round2_masking":    0.0,
        "round3_correction": 0.0,
    }

    for _ in range(n_repeat):
        t0 = time.perf_counter()
        if use_mlkem:
            from secagg.crypto_mlkem import SecAggregatorMLKEM
            clients = {
                sid: SecAggregatorMLKEM(shape=grad_shape, security_level=crypto_backend)
                for sid in sids
            }
        else:
            from secagg.crypto import SecAggregator
            clients = {sid: SecAggregator(shape=grad_shape) for sid in sids}
        totals["round0_keygen"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if use_mlkem:
            peer_eks = {sid: clients[sid].public_key for sid in sids}
            all_cts: Dict = {}
            for sid in sids:
                all_cts[sid] = clients[sid].generate_ciphertexts(peer_eks, sid)
            for i, sid_v in enumerate(sids):
                incoming = {
                    sid_u: all_cts[sid_u][sid_v]
                    for sid_u in sids[:i]
                    if sid_v in all_cts.get(sid_u, {})
                }
                if incoming:
                    clients[sid_v].receive_ciphertexts(incoming)
        totals["round05_encaps"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if use_mlkem:
            for sid in survivor_sids:
                clients[sid].set_weights(np.ones(grad_shape, dtype=np.float64))
                clients[sid].prepare_masked_gradient()
        else:
            all_pks = {sid: clients[sid].public_key for sid in sids}
            for sid in survivor_sids:
                clients[sid].set_weights(np.ones(grad_shape, dtype=np.float64))
                clients[sid].prepare_masked_gradient(all_pks, sid)
        totals["round2_masking"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for sid in survivor_sids:
            clients[sid].reveal_pairwise_masks(dropout_sids)
        totals["round3_correction"] += time.perf_counter() - t0

    return {k: v / n_repeat for k, v in totals.items()}



