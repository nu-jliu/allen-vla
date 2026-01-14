# Udev Rules for SO101 Robot Arms

This directory contains udev rules to automatically configure permissions and create consistent device names for the SO101 robot arms.

## Device Information

- **Vendor**: QinHeng Electronics
- **Vendor ID**: `1a86`
- **Product ID**: `55d3`
- **Description**: USB Single Serial

## Benefits

- Automatic permissions configuration (no need for `sudo`)
- Optional: Consistent device names (`/dev/so101_leader` and `/dev/so101_follower`)
- Devices accessible to users in the `dialout` group

## Quick Start Installation

For basic permissions setup (recommended to start):

### 1. Install the Udev Rule

```bash
sudo cp 99-so101.rules /etc/udev/rules.d/
```

### 2. Add Your User to the dialout Group

```bash
sudo usermod -a -G dialout $USER
```

**Important**: Log out and log back in for the group change to take effect.

### 3. Reload Udev Rules

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4. Test

Disconnect and reconnect your SO101 arms. You should now be able to access them without `sudo`:

```bash
python teleop/main.py
```

## Advanced: Creating Consistent Device Names

If you want to create symbolic links like `/dev/so101_leader` and `/dev/so101_follower`, you need to distinguish between your two arms.

### Option A: Using Serial Numbers

1. **Find serial numbers** for each arm:
   ```bash
   # Connect only the leader arm
   udevadm info -a -n /dev/ttyACM0 | grep serial

   # Connect only the follower arm
   udevadm info -a -n /dev/ttyACM1 | grep serial
   ```

2. **Edit `99-so101.rules`** and uncomment Method 1, replacing `LEADER_SERIAL` and `FOLLOWER_SERIAL` with the actual serial numbers.

3. **Reload rules**:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

### Option B: Using USB Port Location

1. **Find USB port paths**:
   ```bash
   # Connect leader arm to a specific USB port
   udevadm info -a -n /dev/ttyACM0 | grep KERNELS | head -1

   # Connect follower arm to a different USB port
   udevadm info -a -n /dev/ttyACM1 | grep KERNELS | head -1
   ```

2. **Edit `99-so101.rules`** and uncomment Method 2, replacing the `KERNELS` values with your actual USB port paths.

3. **Reload rules**:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

4. **Important**: Always plug the leader arm into the same USB port, and the follower into its designated port.

### Verify Symbolic Links

```bash
ls -l /dev/so101_*
```

Expected output:
```
lrwxrwxrwx 1 root root 7 Dec 18 10:00 /dev/so101_follower -> ttyACM0
lrwxrwxrwx 1 root root 7 Dec 18 10:00 /dev/so101_leader -> ttyACM1
```

## Usage

### With Basic Setup (Method 3)

```bash
# Use default ports or specify which ttyACM device is which
python teleop/main.py --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0
```

### With Symbolic Links (Method 1 or 2)

```bash
# Use consistent device names
python teleop/main.py --leader-port /dev/so101_leader --follower-port /dev/so101_follower
```

## Troubleshooting

### Permission denied errors

1. **Check if you're in the dialout group**:
   ```bash
   groups
   ```
   If `dialout` is not listed, you need to log out and log back in.

2. **Check device permissions**:
   ```bash
   ls -l /dev/ttyACM*
   ```
   Should show `crw-rw-rw-` or similar with group `dialout`.

### Symlinks not created

1. **Check device attributes match**:
   ```bash
   udevadm info -a -n /dev/ttyACM0 | grep -E 'idVendor|idProduct|serial|KERNELS'
   ```

2. **Test the rule**:
   ```bash
   sudo udevadm test $(udevadm info -q path -n /dev/ttyACM0)
   ```

3. **Check udev logs**:
   ```bash
   sudo journalctl -f | grep udev
   ```
   Then disconnect and reconnect a device.

### Finding the right ttyACM device

```bash
# List all USB serial devices with details
for dev in /dev/ttyACM*; do
    echo "Device: $dev"
    udevadm info -a -n $dev | grep -E 'idVendor|idProduct' | head -2
    echo ""
done
```

## Removing the Rules

If you need to remove the udev rules:

```bash
sudo rm /etc/udev/rules.d/99-so101.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```
