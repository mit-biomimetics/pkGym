## Issue ticket number and link
Fixes # (issue)
## Describe your changes
Please include a summary of the change, including why you did this, and the desired effect.

## Instructions for reviewers
Indicate anything in particular that you would like a code-reviewer to pay particular attention to.
Indicate steps to actually test code, including CLI instructions if different than usual.
Point out the desired behavior, and not just the "check that this appears" (otherwise the code reviewer will be lazy and just verify what you've already verified).

## Checklist before requesting a review
- [ ] I have linted my code with flake8 (either in your editor, or from the CLI with `flake8 gpugym/gym` and `flake8 gpugym/learning`)*
- [ ] I have performed a self-review of my code
- [ ] I have assigned a reviewer
- [ ] I have added the PR to the project, and tagged with with priority
- [ ] If it is a core feature, I have added tests.

\* flake8 will trigger on several some files that we don't care about, and on specific linting errors that we cannot get rid of for functional reasons. Just take care of the files/changes you've worked on.