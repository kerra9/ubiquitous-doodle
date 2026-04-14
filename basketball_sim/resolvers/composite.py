"""Composite action resolver -- routes actions to the correct Phase 2 resolver.

This is the real resolver that replaces StubResolver. It reads the action
type, builds the context, runs the modifier pipeline, and delegates to
the appropriate resolver function (dribble, shot, pass, rebound).
"""

from __future__ import annotations

from basketball_sim.core.engine import ActionResolver
from basketball_sim.core.pipeline import ModifierPipeline
from basketball_sim.core.types import (
    Action,
    ActionContext,
    ActionResult,
    ActionType,
    AggregatedModifier,
    EventType,
    GameEvent,
    MatchupState,
    MoveData,
)
from basketball_sim.resolvers.dribble import resolve_dribble
from basketball_sim.resolvers.pass_action import resolve_pass
from basketball_sim.resolvers.rebound import resolve_rebound
from basketball_sim.resolvers.shoot import resolve_shot


class CompositeResolver(ActionResolver):
    """Routes actions to the correct Phase 2 resolver.

    Applies the modifier pipeline before delegating to the specific
    resolver function. Handles all action types defined in ActionType.
    """

    def __init__(
        self,
        pipeline: ModifierPipeline,
        move_registry: dict[str, MoveData] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._moves = move_registry or {}

    def resolve(
        self,
        action: Action,
        matchup: MatchupState,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve an action by routing to the appropriate resolver."""
        # Run the modifier pipeline
        agg = self._pipeline.apply(context)

        if action.action_type == ActionType.DRIBBLE_MOVE:
            return self._resolve_dribble(action, matchup, agg, context)
        elif action.action_type == ActionType.SHOT:
            return resolve_shot(matchup, agg, context)
        elif action.action_type == ActionType.PASS:
            return resolve_pass(matchup, agg, context)
        elif action.action_type == ActionType.DRIVE:
            return self._resolve_drive(action, matchup, agg, context)
        elif action.action_type == ActionType.REBOUND:
            return resolve_rebound(context.possession, context.rng)
        elif action.action_type == ActionType.FREE_THROW:
            return self._resolve_free_throw(action, context)
        else:
            # Fallback: treat as a turnover
            return ActionResult(
                events=[
                    GameEvent(
                        event_type=EventType.TURNOVER,
                        player_id=action.player_id,
                        data={"cause": "unhandled_action", "action_type": action.action_type.name},
                        tags=["turnover"],
                    )
                ],
                ends_possession=True,
            )

    def _resolve_dribble(
        self,
        action: Action,
        matchup: MatchupState,
        agg: AggregatedModifier,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve a dribble move using the move registry."""
        move_id = action.data.get("move", "crossover")
        move = self._moves.get(move_id)

        if move is None:
            # Unknown move -- fall back to a generic dribble
            return ActionResult(
                events=[
                    GameEvent(
                        event_type=EventType.DRIBBLE_MOVE,
                        player_id=action.player_id,
                        data={"move": move_id},
                        tags=["dribble_move"],
                    )
                ],
                tags=["dribble_move"],
                new_matchup=matchup,
            )

        return resolve_dribble(move, matchup, agg, context)

    def _resolve_drive(
        self,
        action: Action,
        matchup: MatchupState,
        agg: AggregatedModifier,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve a drive to the basket.

        A drive is essentially a movement action that transitions into
        a finishing attempt. We resolve it as a shot at the rim.
        """
        rng = context.rng

        # Drive changes the shot type to a layup/dunk attempt
        player = context.attacker
        if player.attributes.dunk >= 75 and rng.random() < 0.35:
            shot_type = "dunk"
        else:
            shot_type = "driving_layup"

        # Modify the action data for the shot resolver
        drive_context = ActionContext(
            action=Action(
                action_type=ActionType.SHOT,
                player_id=action.player_id,
                data={"shot_type": shot_type, "from_drive": True},
                time_cost=action.time_cost,
            ),
            attacker=context.attacker,
            defender=context.defender,
            matchup=matchup,
            possession=context.possession,
            game_state=context.game_state,
            rng=rng,
            cell="D2",  # drives go toward the basket area
        )

        result = resolve_shot(matchup, agg, drive_context)

        # Add drive-specific tags
        if "shot_made" in result.tags:
            result.tags.append("drive_and_finish")
        else:
            result.tags.append("drive_and_miss")

        return result

    def _resolve_free_throw(
        self,
        action: Action,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve a free throw attempt."""
        rng = context.rng
        # Base FT% from mid-range attribute (simplified)
        ft_pct = context.attacker.attributes.mid_range / 120.0  # 70 rating -> 0.583
        ft_pct = max(0.40, min(0.95, ft_pct))

        made = rng.random() < ft_pct
        tags = ["free_throw"]
        if made:
            tags.append("ft_made")
        else:
            tags.append("ft_missed")

        return ActionResult(
            events=[
                GameEvent(
                    event_type=EventType.FREE_THROW,
                    player_id=action.player_id,
                    data={"made": made, "points": 1 if made else 0},
                    tags=tags,
                )
            ],
            tags=tags,
            ends_possession=False,
            score_change=1 if made else 0,
        )
