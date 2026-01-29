# Contributing to Triton VM

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued.
Please make sure to read the relevant section below before making your contribution.
It will make it a lot easier for us maintainers and smooth out the experience for all involved.
The community looks forward to your contributions. 🎉

## I Have a Question

> Please read (or at least scan) the [specification](https://triton-vm.org/spec/) before asking a question.

Before you ask a question, it is best to search for existing [Issues](https://github.com/TritonVM/triton-vm/issues) that might help you.
In case you have found a suitable issue and still need clarification, you can write your question in this issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/TritonVM/triton-vm/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions, depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want to Contribute

### Legal Notice

Any contribution intentionally submitted for inclusion in this repository, as defined in the
Apache-2.0 license, shall be dual licensed under the [MIT](LICENSE-MIT) and
[Apache 2.0](LICENSE-APACHE) licenses, without any additional terms or conditions.

### Reporting Bugs

#### Before Submitting a Bug Report

Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions.
  If you are looking for support, you might want to check section [“I have a question”](#i-have-a-question).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/TritonVM/triton-vm/issues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version of both Triton VM and `rustc`
  - Possibly your input and the output
  - Can you reliably reproduce the issue, or does it happen only occasionally?

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public;
> please refer to [`SECURITY.md`](SECURITY.md) for such cases.

We use GitHub issues to track bugs and errors.
If you run into an issue with the project:

- Open an [Issue](https://github.com/TritonVM/triton-vm/issues/new).
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the _reproduction steps_ that someone else can follow to recreate the issue on their own.
  This usually includes your code.
  For an excellent bug report, create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps.
  If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps.
  A bug will only be addressed once it can be reproduced.
- If the team is able to reproduce the issue, the issue will be left to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Triton VM, **including completely new features and minor improvements to existing functionality**.
Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://triton-vm.org/spec/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/TritonVM/triton-vm/issues) to see if the enhancement has already been suggested.
  If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project.
  It's up to you to make a strong case to convince the project's developers of the merits of this feature.
  Keep in mind that we want features that will be useful to the majority of our users and not just a small subset.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/TritonVM/triton-vm/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
  At this point you can also tell which alternatives do not work for you.
- **Explain why this enhancement would be useful** to most Triton VM users.
  You may also want to point out the other projects that solved it better and which could serve as inspiration.

### Your First Code Contribution

Triton VM uses a pretty standard [`cargo`](https://doc.rust-lang.org/cargo/index.html) setup.
Verify that `cargo test` runs successfully, then dive in.
If you don't already know where you want to start, you might find the [examples](https://github.com/TritonVM/triton-vm/tree/master/triton-vm/examples) helpful.

### Improving the Documentation

Documentation can always be better, and getting into a new project often reveals documentation gaps other developers have gotten blind to.
We welcome documentation updates and upgrades.

### Regarding Low-Effort Contributions

Contributions that are of very low effort will either be incorporated without attributing the contributor or ignored outright.
For example, we consider fixing the spelling of a single word in some documentation file as “low effort.”

The reason for this stance is the following:
Sometimes, cryptocurrencies chose to award some of their tokens to contributors of their ecosystem, of which Triton VM is sometimes a part.
Usually, contributor identification is automated in some fashion, and no regard is given to contribution quality.
(How would you even quantify the quality?)
This fact is used by some people to try to stake a claim at such potential future rewards by getting their name into the contributors list.
We think this is unethical, and therefore, we reject low-effort contributions, or incorporate them in a way that is difficult to automatically link to the original author.

Please do not take above explanation as any form of guarantee that meaningfully contributing to Triton VM will grant you such rewards.
It might be the case, but is both out of our hands and a poor motivator for working on Triton VM.

## Styleguides

### Code

Triton VM uses an [import granularity](https://rust-lang.github.io/rustfmt/#imports_granularity) of `item` and a [group order strategy](https://rust-lang.github.io/rustfmt/#group_imports) of `StdExternalCrate` in order to trivialize git merging of imports.
Unfortunately, both of these `cargo_fmt` features are unstable for now and thus require manual editing or a formatting pass with nightly rust.

Other than that, Triton VM has no formal style requirements at the moment.
However, the core contributors might impose their (informal) style requirements on your pull request. 😊

### Commit Messages

Triton VM uses [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).
We have a [list of established commit types](https://github.com/TritonVM/triton-vm/blob/master/cliff.toml).
If you're missing a commit type, you can introduce it by adding to the list.

In addition to being [conventional](https://www.conventionalcommits.org/en/v1.0.0/), commit messages for Triton VM follow [this guide](https://cbea.ms/git-commit/).
Summarizing:
1. Separate subject from body with a blank line
1. Limit the subject line to 50 characters
1. Capitalize the subject line
1. Do not end the subject line with a period
1. Use the imperative mood in the subject line
1. Wrap the body at 72 characters
1. Use the body to explain _what_ and _why_ vs. _how_

## Attribution

This guide is based on the [contributing.md](https://contributing.md/generator).
