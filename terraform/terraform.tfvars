###############################################################################
# terraform.tfvars
###############################################################################

provider_endpoint   = "https://rancher.condenser.arc.ucl.ac.uk/k8s/clusters/c-m-bv9x5ngh"
provider_token      = "kubeconfig-u-fhgdi4zayzt44cf:vpnbspq6drps267tt2cb2vm4s2l5mk6xxh79fjcvmx77g86mr98pj4"
provider_namespace  = "ucabbaa-comp0235-ns"
username            = "ucabbaa@ucl.ac.uk"

ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDEGYMIHQdWQv/nwFf7zN2Pue60T+Outmr202vz3/DPAA+ufAWfo9FRw4r4dR44jFZpqmwIm2GCIH/JmB9/ETJsYgOJLBhxC+1Dea75jsQh7deYIcyfJQKJpnb0S5FJbg2+6H3jfGLj/MvoU/tVVqz2lNOmpApSj7B0Npo/qkea/3kzXhFz8Bhu4K1Glr0bh1d4YYLQ20SrSpFR+uGcwkKuOBMX12T25tNPADQ+Wi/nVK7jD9124oqcpgCGLDHWpSULX/AYCxCqMQOih3Kb2B6x7OPnCxXPpQugAOJ7zclfdRVpVN07RAPJKjMpVeQ+87EEH5ZFWpuOortUrrGE8xac+OsDOzQzmNCkchcq6rhs3YuPb/U86a6RswimkOGl/J2wT4Dd9npnCtkNyhFKc1Ucr+z63qQ/Pc0binfmgQ9XpX6A0FdMHs0d2XQzgKMjtDVfEEfclVO9ieGUMziDDdNzSce9xeAKWFyXZGXX3kBvmYDJOcs0HRjggUQXhmrHmbff6hkCd8qn2yoTIHtBgSH8VCR0tbT8UEyJFmqrBRt5qdDk4ITiPomdxOcEmdG9ivPtcDgcvP5RMNwUSBbEMWPGn7SHv36Glzm0t4Z4PPZouLr8pnrBzR9xUrd/5soyul0XkkcRGeA9Nj5ChtSsTMZbK26sk3UQjwNCm3+arAouSw== abdullah.alotaishan@kcl.ac.uk"
ssh_key_marker = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMd6wccFYaxf5Mn/Hk5MyRQvugd+FJuWJnvLt8wecr7S dbuchan@ML-RJKH0G50C0.local"

ssh_ansible_private = <<EOF
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACANrYXDStWTCsG+avATz1DkQr8JzBamqs+FsQRBgYQ6jAAAAKgOOf5BDjn+
QQAAAAtzc2gtZWQyNTUxOQAAACANrYXDStWTCsG+avATz1DkQr8JzBamqs+FsQRBgYQ6jA
AAAEBm6pD/vmBuAOhgPpePHzHe7wm8CgtIm4SJepTeLmOvsw2thcNK1ZMKwb5q8BPPUORC
vwnMFqaqz4WxBEGBhDqMAAAAJHJvb3RAdWNhYmJhYS11Y2wtYWMtdWstbWdtdC1kMTI1ZG
ViMgE=
-----END OPENSSH PRIVATE KEY-----
EOF

ssh_ansible_public  = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA2thcNK1ZMKwb5q8BPPUORCvwnMFqaqz4WxBEGBhDqM root@ucabbaa-ucl-ac-uk-mgmt-d125deb2"

image_name      = "image-bp52g"
image_namespace = "harvester-public"
network_name    = "ucabbaa-comp0235-ns/ds4eng"

# VM Specs
mgmt_cpu       = 2
mgmt_memory    = "4Gi"
mgmt_disk_size = "10Gi"

worker_count     = 3
worker_cpu       = 4
worker_memory    = "32Gi"
worker_disk_size = "25Gi"

# Storage VM specs
storage_cpu             = 4
storage_memory          = "8Gi"
storage_root_disk_size  = "10Gi"
storage_extra_disk_size = "200Gi"
