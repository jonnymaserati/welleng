# Security Policy

## Supported versions

welleng is pre-1.0 and released continuously. Security fixes are applied to the
**latest release** on [PyPI](https://pypi.org/project/welleng/); please upgrade to
the latest version before reporting.

| Version | Supported |
| ------- | --------- |
| latest release | :white_check_mark: |
| older releases | :x: |

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability.

Report it privately using GitHub's **[Report a vulnerability](https://github.com/jonnymaserati/welleng/security/advisories/new)**
flow (the *Security* tab → *Report a vulnerability*), or by email to
**jonnycorcutt@gmail.com**.

Please include, as far as you can:

- a description of the vulnerability and its impact,
- the affected version(s),
- steps or a minimal example to reproduce it.

You can expect an acknowledgement of your report, and we will keep you informed as
we assess and address it. Once a fix is released, we are happy to credit you for the
disclosure unless you prefer to remain anonymous.

## Scope

welleng is a scientific Python library for well-engineering calculations. It runs
locally as an imported package — it does not provide a network service, handle
authentication, or process untrusted credentials. The most relevant classes of issue
are therefore things like unsafe handling of untrusted input files (e.g. survey /
error-model imports) or vulnerabilities introduced via a dependency.
