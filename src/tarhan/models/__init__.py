"""TARHAN Layer-3 domain modelleri.

Mimari sözleşme (Product Form Decision + Architecture notu): domain modelleri
İNCE katmandır — tür tanımları, malzeme korelasyonları ve sınır kinetiğini
`tarhan.physics` (oracle-verified formüller) ve `tarhan.numerics` üzerine kurar;
çekirdek koda dokunmaz (zero-core-edit). Modeller: sofc1d (O'Hayre §6.2),
pemfc0d (Kim-1995/Barbir kayıp-merdiveni), pn1d (Gummel/SG drift-diffusion).
"""
