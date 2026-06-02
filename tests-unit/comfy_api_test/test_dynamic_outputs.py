"""Unit tests for ``DynamicOutputs.ByKey`` and the finalized-outputs path."""

import pytest

from comfy_api.latest import _io as io


# ---------------------------------------------------------------------------
# Schema-level construction and validation
# ---------------------------------------------------------------------------

def _byke():
    return io.DynamicOutputs.ByKey(
        id="result",
        selector="mode",
        options=[
            io.DynamicOutputs.Option(key="image",
                                     outputs=[io.Image.Output("image"), io.Mask.Output("mask")]),
            io.DynamicOutputs.Option(key="latent",
                                     outputs=[io.Latent.Output("latent")]),
        ],
    )


def test_option_rejects_empty_key():
    with pytest.raises(ValueError, match="non-empty string"):
        io.DynamicOutputs.Option(key="", outputs=[])


def test_option_rejects_non_output_entry():
    with pytest.raises(ValueError, match="Output instances"):
        io.DynamicOutputs.Option(key="x", outputs=["not an output"])


def test_option_requires_explicit_output_ids():
    with pytest.raises(ValueError, match="declare an id"):
        io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output()])  # no id


def test_bykey_rejects_empty_options():
    with pytest.raises(ValueError, match="at least one Option"):
        io.DynamicOutputs.ByKey(id="r", selector="m", options=[])


def test_bykey_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate option key"):
        io.DynamicOutputs.ByKey(
            id="r", selector="m",
            options=[
                io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output("a")]),
                io.DynamicOutputs.Option(key="x", outputs=[io.Latent.Output("b")]),
            ],
        )


def test_bykey_rejects_duplicate_output_ids_across_options():
    with pytest.raises(ValueError, match="appears in more than one option"):
        io.DynamicOutputs.ByKey(
            id="r", selector="m",
            options=[
                io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output("dup")]),
                io.DynamicOutputs.Option(key="y", outputs=[io.Latent.Output("dup")]),
            ],
        )


# ---------------------------------------------------------------------------
# Schema integration
# ---------------------------------------------------------------------------

def _make_node(extra_outputs=None):
    """Build a V3 node class with a selector input + DynamicOutputs group."""
    extras = extra_outputs or []

    class DynNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DynNode",
                inputs=[io.Combo.Input("mode", options=["image", "latent"], default="image")],
                outputs=[*extras, _byke()],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"image": None, "mask": None})

    return DynNode


def test_schema_validate_rejects_unknown_selector():
    class BadSelector(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="BadSelector",
                inputs=[io.Combo.Input("not_mode", options=["a"])],
                outputs=[
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Image.Output("a")])],
                    ),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"a": None})

    with pytest.raises(ValueError, match="selector input 'mode' does not exist"):
        BadSelector.GET_SCHEMA()


def test_schema_validate_rejects_id_collision_with_static_output():
    class Collision(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Collision",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[
                    io.Image.Output("shared"),
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Latent.Output("shared")])],
                    ),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"shared": None})

    with pytest.raises(ValueError, match="Output ids must be unique"):
        Collision.GET_SCHEMA()


def test_schema_get_v1_info_emits_dynamic_outputs_field():
    DynNode = _make_node()
    DynNode.GET_SCHEMA()
    info = DynNode.SCHEMA.get_v1_info(DynNode)
    assert info.dynamic_outputs is not None and len(info.dynamic_outputs) == 1
    group = info.dynamic_outputs[0]
    assert group["kind"] == "by_key"
    assert group["selector"] == "mode"
    assert {opt["key"] for opt in group["options"]} == {"image", "latent"}
    # Static output arrays are empty — only the dynamic group is declared.
    assert info.output == []
    assert info.output_is_list == []


def test_schema_static_outputs_stable_prefix_in_v1_arrays():
    """A static output before a dynamic group still surfaces in RETURN_TYPES etc."""
    DynNode = _make_node(extra_outputs=[io.String.Output("status")])
    DynNode.GET_SCHEMA()
    # Class-level static arrays are the always-present prefix.
    assert list(DynNode.RETURN_TYPES) == ["STRING"]
    assert list(DynNode.RETURN_NAMES) == ["status"]
    assert list(DynNode.OUTPUT_IS_LIST) == [False]


# ---------------------------------------------------------------------------
# get_finalized_class_outputs
# ---------------------------------------------------------------------------

def test_finalize_picks_active_branch():
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "latent"})
    assert finalized.output_ids == ["latent"]
    assert finalized.return_types == ["LATENT"]
    assert finalized.output_is_list == [False]


def test_finalize_unknown_selector_yields_empty():
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "nonexistent"})
    assert len(finalized) == 0


def test_finalize_link_selector_yields_empty():
    """Link as selector value is treated as 'not finalizable' — no branch."""
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": ["src", 0]})
    assert len(finalized) == 0


def test_finalize_static_prefix_preserved():
    schema_outputs = [io.String.Output("status"), _byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "image"})
    assert finalized.output_ids == ["status", "image", "mask"]
    assert finalized.return_types == ["STRING", "IMAGE", "MASK"]


# ---------------------------------------------------------------------------
# NodeOutput.from_named
# ---------------------------------------------------------------------------

def test_nodeoutput_from_named_stores_dict():
    out = io.NodeOutput.from_named({"a": 1, "b": 2})
    assert out.named == {"a": 1, "b": 2}
    assert out.args == ()
    assert out.result is None  # `.result` is the positional tuple


def test_nodeoutput_rejects_mixed_positional_and_named():
    with pytest.raises(ValueError, match="cannot mix positional"):
        io.NodeOutput(1, 2, named={"a": 1})


# ---------------------------------------------------------------------------
# Group-id uniqueness
# ---------------------------------------------------------------------------

def test_schema_rejects_duplicate_dynamic_group_ids():
    class Dup(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Dup",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[
                    io.DynamicOutputs.ByKey(id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Image.Output("x")])]),
                    io.DynamicOutputs.ByKey(id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Latent.Output("y")])]),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"x": None, "y": None})

    with pytest.raises(ValueError, match="DynamicOutputs group ids must be unique"):
        Dup.GET_SCHEMA()


# ---------------------------------------------------------------------------
# DynamicOutputs.ByKey with a DynamicCombo selector
# ---------------------------------------------------------------------------

def _combo_input():
    return io.DynamicCombo.Input("mode", options=[
        io.DynamicCombo.Option(key="image", inputs=[io.Image.Input("img")]),
        io.DynamicCombo.Option(key="latent", inputs=[io.Latent.Input("lat")]),
    ])


def _bykey_outputs():
    return io.DynamicOutputs.ByKey(id="result", selector="mode", options=[
        io.DynamicOutputs.Option(key="image", outputs=[io.Image.Output("processed"), io.Mask.Output("alpha")]),
        io.DynamicOutputs.Option(key="latent", outputs=[io.Latent.Output("denoised")]),
    ])


def test_bykey_with_dynamic_combo_finalizes_branch():
    finalized = io.get_finalized_class_outputs(
        [io.String.Output("status"), _bykey_outputs()],
        {"mode": {"mode": "image", "img": None}},  # DynamicCombo dispatch shape
    )
    assert finalized.output_ids == ["status", "processed", "alpha"]
    assert finalized.return_types == ["STRING", "IMAGE", "MASK"]


def test_bykey_with_dynamic_combo_other_branch():
    finalized = io.get_finalized_class_outputs(
        [_bykey_outputs()],
        {"mode": {"mode": "latent", "lat": None}},
    )
    assert finalized.output_ids == ["denoised"]


def test_schema_rejects_bykey_key_not_on_dynamic_combo():
    class StrayKey(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="StrayKey",
                inputs=[_combo_input()],
                outputs=[io.DynamicOutputs.ByKey(id="r", selector="mode", options=[
                    io.DynamicOutputs.Option(key="image", outputs=[io.Image.Output("a")]),
                    io.DynamicOutputs.Option(key="audio", outputs=[io.String.Output("b")]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match=r"option key\(s\) \['audio'\] are not declared"):
        StrayKey.GET_SCHEMA()


# ---------------------------------------------------------------------------
# DynamicOutputs.BySlot
# ---------------------------------------------------------------------------

def _slot_input():
    return io.DynamicSlot.Input("slot", options=[
        io.DynamicSlot.Option(when=io.Image),
        io.DynamicSlot.Option(when=io.Latent),
        io.DynamicSlot.Option(when=None, inputs=[io.Int.Input("seed")]),
    ])


def _byslot_outputs():
    return io.DynamicOutputs.BySlot(id="slot_out", selector="slot", options=[
        io.DynamicOutputs.SlotOption(when=io.Image, outputs=[io.Image.Output("processed"), io.Mask.Output("alpha")]),
        io.DynamicOutputs.SlotOption(when=io.Latent, outputs=[io.Latent.Output("denoised")]),
        io.DynamicOutputs.SlotOption(when=None, outputs=[]),
    ])


def test_byslot_finalizes_by_resolved_type():
    finalized = io.get_finalized_class_outputs(
        [_byslot_outputs()],
        {"slot": ["upstream", 0]},
        live_input_types={"slot": "IMAGE"},
    )
    assert finalized.output_ids == ["processed", "alpha"]
    finalized = io.get_finalized_class_outputs(
        [_byslot_outputs()],
        {"slot": ["upstream", 0]},
        live_input_types={"slot": "LATENT"},
    )
    assert finalized.output_ids == ["denoised"]


def test_byslot_unconnected_uses_when_none():
    finalized = io.get_finalized_class_outputs([_byslot_outputs()], {})
    # when=None option declares outputs=[] → no active outputs
    assert finalized.output_ids == []


def test_byslot_unmatched_type_yields_empty():
    finalized = io.get_finalized_class_outputs(
        [_byslot_outputs()],
        {"slot": ["upstream", 0]},
        live_input_types={"slot": "AUDIO"},
    )
    assert finalized.output_ids == []


def test_byslot_rejects_duplicate_when_types():
    with pytest.raises(ValueError, match="appears in more than one option"):
        io.DynamicOutputs.BySlot(id="r", selector="slot", options=[
            io.DynamicOutputs.SlotOption(when=io.Image, outputs=[io.Image.Output("a")]),
            io.DynamicOutputs.SlotOption(when=io.Image, outputs=[io.Mask.Output("b")]),
        ])


def test_byslot_rejects_duplicate_when_none():
    with pytest.raises(ValueError, match="only one option may declare when=None"):
        io.DynamicOutputs.BySlot(id="r", selector="slot", options=[
            io.DynamicOutputs.SlotOption(when=None, outputs=[]),
            io.DynamicOutputs.SlotOption(when=None, outputs=[io.Image.Output("x")]),
        ])


def test_schema_rejects_byslot_selector_not_a_dynamic_slot():
    class WrongSel(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="WrongSel",
                inputs=[io.Combo.Input("not_a_slot", options=["a"])],
                outputs=[io.DynamicOutputs.BySlot(id="r", selector="not_a_slot", options=[
                    io.DynamicOutputs.SlotOption(when=io.Image, outputs=[io.Image.Output("x")]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match="must reference a DynamicSlot input"):
        WrongSel.GET_SCHEMA()


def test_schema_rejects_byslot_when_type_not_on_slot():
    class StrayWhen(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="StrayWhen",
                inputs=[_slot_input()],
                outputs=[io.DynamicOutputs.BySlot(id="r", selector="slot", options=[
                    io.DynamicOutputs.SlotOption(when=io.Audio, outputs=[io.Audio.Output("x")]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match=r"type\(s\) \['AUDIO'\] are not accepted"):
        StrayWhen.GET_SCHEMA()


def test_schema_rejects_byslot_when_none_without_slot_when_none():
    class NoNone(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="NoNone",
                inputs=[io.DynamicSlot.Input("slot", optional=False, options=[
                    io.DynamicSlot.Option(when=io.Image),
                ])],
                outputs=[io.DynamicOutputs.BySlot(id="r", selector="slot", options=[
                    io.DynamicOutputs.SlotOption(when=None, outputs=[]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match="requires DynamicSlot 'slot' to declare a when=None"):
        NoNone.GET_SCHEMA()


def test_v1_info_emits_byslot_entry():
    class N(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="SlotV1",
                inputs=[_slot_input()],
                outputs=[_byslot_outputs()],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    N.GET_SCHEMA()
    info = N.SCHEMA.get_v1_info(N)
    assert info.dynamic_outputs is not None and len(info.dynamic_outputs) == 1
    entry = info.dynamic_outputs[0]
    assert entry["kind"] == "by_slot"
    assert entry["selector"] == "slot"
    whens = [opt["when"] for opt in entry["options"]]
    assert whens == [["IMAGE"], ["LATENT"], None]


# ---------------------------------------------------------------------------
# Nested dynamic selectors (selector reaches into a DynamicCombo / DynamicSlot
# option's nested inputs via a dotted path).
# ---------------------------------------------------------------------------

def test_schema_accepts_byslot_selector_into_nested_dynamic_slot():
    """A BySlot selector may target a DynamicSlot nested inside a DynamicCombo option."""
    class Nested(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="NestedBySlot",
                inputs=[
                    io.DynamicCombo.Input("outer", options=[
                        io.DynamicCombo.Option(key="a", inputs=[
                            io.DynamicSlot.Input("inner_slot", options=[
                                io.DynamicSlot.Option(when=io.Image),
                                io.DynamicSlot.Option(when=None),
                            ]),
                        ]),
                    ]),
                ],
                outputs=[io.DynamicOutputs.BySlot(id="r", selector="outer.inner_slot", options=[
                    io.DynamicOutputs.SlotOption(when=io.Image, outputs=[io.Image.Output("img")]),
                    io.DynamicOutputs.SlotOption(when=None, outputs=[]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    # Must not raise — selector resolves to the nested DynamicSlot via dotted path.
    Nested.GET_SCHEMA()


def test_schema_accepts_bykey_selector_into_nested_dynamic_combo():
    """A ByKey selector may target a DynamicCombo nested inside another DynamicCombo option."""
    class Nested(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="NestedByKey",
                inputs=[
                    io.DynamicCombo.Input("outer", options=[
                        io.DynamicCombo.Option(key="branch", inputs=[
                            io.DynamicCombo.Input("inner", options=[
                                io.DynamicCombo.Option(key="x", inputs=[]),
                                io.DynamicCombo.Option(key="y", inputs=[]),
                            ]),
                        ]),
                    ]),
                ],
                outputs=[io.DynamicOutputs.ByKey(id="r", selector="outer.inner", options=[
                    io.DynamicOutputs.Option(key="x", outputs=[io.Int.Output("ix")]),
                    io.DynamicOutputs.Option(key="y", outputs=[io.String.Output("iy")]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    Nested.GET_SCHEMA()


def test_schema_rejects_byslot_nested_when_type_not_on_target_slot():
    """The when-type alignment check follows the dotted selector into the nested slot."""
    class BadNested(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="BadNested",
                inputs=[
                    io.DynamicCombo.Input("outer", options=[
                        io.DynamicCombo.Option(key="a", inputs=[
                            io.DynamicSlot.Input("inner_slot", options=[
                                io.DynamicSlot.Option(when=io.Image),
                            ]),
                        ]),
                    ]),
                ],
                outputs=[io.DynamicOutputs.BySlot(id="r", selector="outer.inner_slot", options=[
                    io.DynamicOutputs.SlotOption(when=io.Audio, outputs=[io.Audio.Output("x")]),
                ])],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match=r"type\(s\) \['AUDIO'\] are not accepted"):
        BadNested.GET_SCHEMA()
