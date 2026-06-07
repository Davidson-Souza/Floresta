# Installing Floresta

If you prefer skipping the [build-unix](./build-unix.md) and [run](./run.md) steps to directly install our binaries you have the below options.

1. If you have [`just`](https://just.systems/) you can simply do `just install` to install both `florestad` and `floresta-cli` or you can specify one of them to install only the specified. Refer to the [command's docs written in the justfile](../justfile) if you clarification about the process.

or if you prefer a guided installation process, that will help you to configure your daemon with your own needs:

2. with this  [`bash` script to install and configure](../contrib/install.sh).

> ⚠️ Currently, this script only works for debian-based systems. Also it aims to run on fresh servers, since it does some package manager operations, that is `cargo` and `apt` removing or adding dependencies. If you are insecure about the process is strongly advised to understand how the script works.

## About the script

This script will guide the user about Floresta's installation/configuration or removal from the system. In the case of install, it will:

* check if you have the required dependencies to build Floresta, and if not, download them;
* start the guide to desired Floresta's `systemd` service configuration;
* once configured, download the [latest release](https://github.com/getfloresta/Floresta/releases/latest), build and install the `florestad` and `floresta-cli` binaries with the `jsonrpc` feature;
* warn how to start and enable the `systemd` service;
* check if the service is running.

## Running the script

Use `curl` to download the script and execute it with `bash`:

```bash
curl -LsSf https://raw.githubusercontent.com/getfloresta/Floresta/refs/heads/master/contrib/install.sh | bash
```

Or you can use `wget`:

```bash
wget -qO- https://raw.githubusercontent.com/getfloresta/Floresta/refs/heads/master/contrib/install.sh | bash
```

### Check dependencies

When you start the command above you could see something like this:

```bash
$ curl -LsSf https://raw.githubusercontent.com/getfloresta/Floresta/refs/heads/master/contrib/install.sh | bash
✅ Internet connection is available.
🐧 'dialog' is not installed. Installing...
[sudo] password for user:
```

The `'dialog' is not installed. Installing...` message will not appear if you already have `dialog` utility installed. It will be used
to guide you through the setup.

### Start configuration

After the dependencies pre-installation you will see this prompt:

![install.sh initial message](/doc/assets/install-0.png)

### Configure

Press `Ok` to proceed, either to install or uninstall:

![install or uninstall](/doc/assets/install-1.png)

Then you could choose between some basic options, advanced options and review configuration:

![Main menu](/doc/assets/install-2.png)

Choose a supported network for your node:

![choose network](/doc/assets/install-3.png)

You can also add any master public-key, descriptors or specific addresses:

![add wallet](/doc/assets/install-4.png)

The script also have advanced setup. They are not required, but some are desirable to have, like TLS:

![advanced config](/doc/assets/install-5.png)

### Review

Before install, it is important to review your configuration:

![review](/doc/assets/install-6.png)

### Running

After the installation procedure finishes, you'll see this message:

```bash
✅ DONE

⚠️ Before enable/start, please edit '/usr/lib/systemd/system/florestad.service' to your needs. After that, run:

    sudo systemctl daemon-restart           # restart daemon definitions
    sudo systemctl start florestad.service  # this will start the service now
    sudo systemctl status florestad.service # this check if service is running well
    sudo systemctl enable florestad.service # this enable service on boot
    sudo -u florestad floresta-cli --network bitcoin --data-dir /var/lib/florestad getblockchaininfo

The RPC cookie is owned by the 'florestad' user and readable only by it,
so run floresta-cli as that user, or set -rpcuser/-rpcpassword instead.
```

This means that both `florestad` and `floresta-cli` are installed. You could check with:

```bash
which florestad
```

and

```bash
which floresta-cli
```

### Troubleshooting

#### Check the `florestad.service`

Check the file `/usr/lib/systemd/system/florestad.service`. It is optional, but could be worth if you want to change anything.

#### Start the service now

This will run the service in background, but it could fail since it might be needed to reboot your system to apply some permission changes.

#### Check if service is running well

Anytime you want to check you service, you may want to use these commands:

```bash
# Journaling
sudo journalctl -xeu florestad.service

# logs when using bitcoin mainnet
sudo cat /var/lib/florestad/debug.log

# logs when using another network (testnet, signet, regtest)
sudo cat /var/lib/florestad/<network>/debug.log
```

#### Electrum

If you plan to use a tls-enabled electrum client in a coordinator (like sparrow), you may want to copy the `certificate` file. If you're using a remote server:

```bash
# on server if using mainnet
cp /var/lib/florestad/tls/cert.pem

# on server if using another network
cp /var/lib/florestad/<network>/tls/cert.pem /home/user/cert.pem

# on you local machine
scp user@mydomain:/home/user/cert.pem .
```
