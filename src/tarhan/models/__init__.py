"""TARHAN Layer-3 domain modelleri.

Mimari sözleşme (Product Form Decision + Architecture notu): domain modelleri
İNCE katmandır — tür tanımları, malzeme korelasyonları ve sınır kinetiğini
`tarhan.physics` (oracle-verified formüller) ve `tarhan.numerics` üzerine kurar;
çekirdek koda dokunmaz (zero-core-edit). Modeller: sofc1d (O'Hayre §6.2),
pemfc0d (Spiegel/FuelCellStore kayıp-merdiveni), pn1d (Gummel/SG drift-diffusion),
chronoamp1d (transient/BDF kronoamperometri — Cottrell'in implicit yolu).
"""
