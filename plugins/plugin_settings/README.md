# Plugin settings

Declarative settings forms for plugins that do not ship their own Web module,
external plugins in particular. A plugin declares its fields in Python and the
host renders them in the **Plugin settings** card under **Settings › Plugins › Plugin config**:

```python
from plugins.plugin_settings.contract import SETTINGS_FORMS, SettingsField, SettingsForm, declare_settings

FORM = SettingsForm(
    title="GitHub Watch", description="...",
    fields=(
        SettingsField("model", "Review model", kind="model", use=REVIEW_MODEL),
        SettingsField("poll_seconds", "Poll interval (seconds)", kind="number"),
        SettingsField("repositories", "Watched repositories", kind="lines"),
    ),
)

# Optional dependency: the child activation only runs while plugin_settings is enabled.
await ctx.inject((SETTINGS_FORMS, PLUGIN_CONFIG, MODEL_CATALOG),
                 lambda child: declare_settings(child, Config, FORM), name="settings")
```

Field kinds are `text`, `lines` (one item per line), `number`, `toggle` and
`model` (the plugin's own model choice, see ADR 0129). Values are read from and
saved to the plugin's own config through `ModelSettings`: an empty value removes
the override and the config model's default applies. Saving takes effect through
the normal plugin config generation change.

A `model` field's `use` only decides which options the picker offers. Config can
also be written elsewhere, so the plugin's config model must validate the field
itself, e.g. `field_validator("model")` calling `REVIEW_MODEL.check(value)`.

Each plugin may register one form. The host Web contract only accepts registrations made synchronously during activation, so all declared forms share one settings card that loads the list when opened; a plugin that needs its own card or family still ships a Web module. Plugins that need richer interaction (OAuth,
previews, custom validation flows) still ship their own Web module.
