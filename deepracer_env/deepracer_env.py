#################################################################################
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          #
#                                                                               #
#   Licensed under the Apache License, Version 2.0 (the "License").             #
#   You may not use this file except in compliance with the License.            #
#   You may obtain a copy of the License at                                     #
#                                                                               #
#       http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                               #
#   Unless required by applicable law or agreed to in writing, software         #
#   distributed under the License is distributed on an "AS IS" BASIS,           #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#   See the License for the specific language governing permissions and         #
#   limitations under the License.                                              #
#################################################################################
"""A class for DeepRacerEnv environment."""
from typing import Dict, Optional, List, Tuple, Any, FrozenSet, Union
import math

from gym import Space

from ude import (
    UDEEnvironmentInterface, AbstractSideChannel,
    UDEEnvironment, RemoteEnvironmentAdapter,
    AgentID, MultiAgentDict, UDEStepResult,
    Compression, ChannelCredentials
)

from deepracer_env_config import (
    Client,
    Track, Agent
)


class DeepRacerEnv(UDEEnvironmentInterface):
    """
    DeepRacerEnv Class.
    """
    def __init__(self,
                 address: str,
                 port: Optional[int] = 80,
                 options: Optional[List[Tuple[str, Any]]] = None,
                 compression: Compression = Compression.NoCompression,
                 credentials: Optional[Union[str, bytes, ChannelCredentials]] = None,
                 auth_key: Optional[str] = None,
                 timeout: float = 10.0,
                 max_retry_attempts: int = 5):
        """
        Initialize RemoteEnvironmentAdapter.

        Args:
            address (str): address of UDE Server
            port (Optional[int]): the port of UDE Server (default: 80)
            options (Optional[List[Tuple[str, Any]]]): An optional list of key-value pairs
                                                        (:term:`channel_arguments` in gRPC runtime)
                                                        to configure the channel.
            compression (Compression) = channel compression type (default: NoCompression)
            credentials: Optional[Union[str, bytes, ChannelCredentials]]: grpc.ChannelCredentials, the path to
                certificate file or bytes of the certificate to use with an SSL-enabled Channel.
            auth_key (Optional[str]): channel authentication key (only applied when credentials are provided).
            timeout (float): the time-out of grpc.io call
            max_retry_attempts (int): maximum number of retry
        """
        adapter = RemoteEnvironmentAdapter(address=address,
                                           port=port,
                                           options=options,
                                           compression=compression,
                                           credentials=credentials,
                                           auth_key=auth_key,
                                           timeout=timeout,
                                           max_retry_attempts=max_retry_attempts)
        self._env = UDEEnvironment(ude_env_adapter=adapter)
        self._deepracer_config = Client(self._env.side_channel,
                                        timeout=timeout,
                                        max_retry_attempts=max_retry_attempts)
        area_config = self._deepracer_config.get_area()
        self._track_names = area_config.track_names
        self._shell_names = area_config.shell_names

    def step(self, action_dict: MultiAgentDict) -> UDEStepResult:
        """
        Performs one multi-agent step with given action, and retrieve
        observation(s), reward(s), done(s), action(s) taken,
        and info (if there is any).

        Args:
            action_dict (MultiAgentDict): the action(s) for the agent(s) with agent_name as key.

        Returns:
            UDEStepResult: observations, rewards, dones, last_actions, info
        """
        try:
            action_dict = {agent_id: (float(action[0]), float(action[1]))
                           for agent_id, action in action_dict.items()}
        except (TypeError, IndexError):
            raise ValueError("Agent's action value must contain two float values.")

        for agent_id, action in action_dict.items():
            steering_angle, speed = float(action[0]), float(action[1])
            if math.isnan(steering_angle) or math.isinf(steering_angle) or \
               math.isnan(speed) or math.isinf(speed):
                raise ValueError("Agent's action value cannot contain nan or inf: {{}: {}}".format(agent_id,
                                                                                                   action))
        return self._env.step(action_dict=action_dict)

    def reset(self) -> MultiAgentDict:
        """
        Reset the environment and start new episode.
        Also, returns the first observation for new episode started.

        Returns:
            MultiAgentDict: first observation in new episode.
        """
        return self._env.reset()

    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        """
        return self._env.close()

    @property
    def observation_space(self) -> Dict[AgentID, Space]:
        """
        Returns the observation spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the observation spaces of agents in env.
        """
        return self._env.observation_space

    @property
    def action_space(self) -> Dict[AgentID, Space]:
        """
        Returns the action spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the action spaces of agents in env.
        """
        return self._env.action_space

    @property
    def side_channel(self) -> AbstractSideChannel:
        """
        Returns side channel to send to and receive data from environment

        Returns:
            AbstractSideChannel: the instance of side channel.
        """
        return self._env.side_channel

    def get_track(self) -> Track:
        """
        Returns the track configuration.

        Returns:
            Track: the track configuration.
        """
        return self._deepracer_config.get_track()

    def apply_track(self, track: Track) -> None:
        """
        Applies given track configuration.

        Args:
            track (Track): the track configuration to set.
        """
        track_name = track.name
        track_name = track_name.lower().strip()
        if track_name not in self.track_names:
            raise ValueError("{} is not a valid track".format(track.name))
        track.name = track_name
        self._deepracer_config.apply_track(track=track)

    def get_agent(self) -> Agent:
        """
        Returns the agent configuration.

        Returns:
            Track: the agent configuration.
        """
        agents = self._deepracer_config.get_agents()
        return agents[0] if agents else agents

    def apply_agent(self, agent: Agent) -> None:
        """
        Applies given agent configuration.

        Args:
            agent (Agent): the agent configuration to set.
        """
        shell = agent.shell
        shell = shell.lower().strip()
        if shell not in self.shell_names:
            raise ValueError("{} is not a valid shell.".format(agent.shell))
        agent.shell = shell
        self._deepracer_config.apply_agent(agent=agent)

    @property
    def track_names(self) -> FrozenSet[str]:
        """
        Returns a frozenset of tracks supported.

        Returns:
            FrozenSet[str]: a frozenset of tracks supported.
        """
        return self._track_names

    @property
    def shell_names(self) -> FrozenSet[str]:
        """
        Returns a frozenset of custom shells supported.

        Returns:
            FrozenSet[str]: a frozenset of custom shells supported.
        """
        return self._shell_names
