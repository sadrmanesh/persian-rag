# Releasing

Releases are driven by tags. Pushing `vX.Y.Z` runs the `Release` workflow:

1. **Build**: checks that the tag matches `version` in `pyproject.toml`, builds the sdist and wheel, and runs `twine check --strict`.
2. **Publish**: uploads to PyPI with one of two methods.
3. **Docker**: builds the image for `linux/amd64` and `linux/arm64` and pushes it to `ghcr.io/sadrmanesh/persian-rag` with the tags `X.Y.Z`, `X.Y`, and `latest`. It signs in with the workflow's `GITHUB_TOKEN`, so no extra secret is needed.

| Method | Setup |
| --- | --- |
| Trusted publishing (recommended) | Add a trusted publisher on PyPI for this repository, workflow `release.yml`, environment `pypi`. Then set the repository variable `PYPI_TRUSTED_PUBLISHING` to `true`. |
| API token | Add a `PYPI_API_TOKEN` repository secret. |

If neither is configured, the build still runs, the publish job prints a notice, and nothing is uploaded.

## Checklist

1. Update `version` in `pyproject.toml` and `src/persian_rag/version.py`.
2. Add a section to `CHANGELOG.md`.
3. Merge to `main` with green CI.
4. Tag and push: `git tag v0.3.1 && git push origin v0.3.1`.
