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
from typing import Any, Callable, Optional, Iterable
from unittest import mock, TestCase
from unittest.mock import patch, MagicMock, call
import inspect

import math

from deepracer_env import DeepRacerEnv
from ude import Compression


myself: Callable[[], Any] = lambda: inspect.stack()[1][3]


@patch("deepracer_env.deepracer_env.Client")
@patch("deepracer_env.deepracer_env.UDEEnvironment")
@patch("deepracer_env.deepracer_env.RemoteEnvironmentAdapter")
class DeepRacerTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_initialize(self,
                        remote_env_adapter_mock,
                        ude_env_mock,
                        deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        remote_env_adapter_mock.assert_called_once_with(address=address,
                                                        port=80,
                                                        options=None,
                                                        compression=Compression.NoCompression,
                                                        credentials=None,
                                                        auth_key=None,
                                                        timeout=10.0,
                                                        max_retry_attempts=5)
        ude_env_mock.assert_called_once_with(ude_env_adapter=remote_env_adapter_mock.return_value)
        deepracer_config_mock.assert_called_once_with(ude_env_mock.return_value.side_channel,
                                                      timeout=10.0,
                                                      max_retry_attempts=5)
        deepracer_config_mock.return_value.get_area.assert_called_once()
        assert env.track_names == deepracer_config_mock.return_value.get_area.return_value.track_names
        assert env.shell_names == deepracer_config_mock.return_value.get_area.return_value.shell_names

    def test_initialize_with_param(self,
                                   remote_env_adapter_mock,
                                   ude_env_mock,
                                   deepracer_config_mock):
        address = "test_ip"
        port = 4040
        options = [("option1", "value")]
        compression = Compression.Deflate
        credentials = MagicMock()
        auth_key = "auth_pass"
        timeout = 15.0
        max_retry_attempts = 3

        env = DeepRacerEnv(address=address,
                           port=port,
                           options=options,
                           compression=compression,
                           credentials=credentials,
                           auth_key=auth_key,
                           timeout=timeout,
                           max_retry_attempts=max_retry_attempts)

        remote_env_adapter_mock.assert_called_once_with(address=address,
                                                        port=port,
                                                        options=options,
                                                        compression=compression,
                                                        credentials=credentials,
                                                        auth_key=auth_key,
                                                        timeout=timeout,
                                                        max_retry_attempts=max_retry_attempts)
        ude_env_mock.assert_called_once_with(ude_env_adapter=remote_env_adapter_mock.return_value)
        deepracer_config_mock.assert_called_once_with(ude_env_mock.return_value.side_channel,
                                                      timeout=timeout,
                                                      max_retry_attempts=max_retry_attempts)
        deepracer_config_mock.return_value.get_area.assert_called_once()
        assert env.track_names == deepracer_config_mock.return_value.get_area.return_value.track_names
        assert env.shell_names == deepracer_config_mock.return_value.get_area.return_value.shell_names

    def test_step(self,
                  remote_env_adapter_mock,
                  ude_env_mock,
                  deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)

        action_dict = {"agent1": (1.0, 2.0)}

        step_result = env.step(action_dict=action_dict)
        ude_env_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        assert step_result == ude_env_mock.return_value.step.return_value

    def test_step_single_value(self,
                               remote_env_adapter_mock,
                               ude_env_mock,
                               deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)

        action_dict = {"agent1": 1.0}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

        action_dict = {"agent1": [1.0]}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

    def test_step_ignore_more_than_two_value(self,
                                             remote_env_adapter_mock,
                                             ude_env_mock,
                                             deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)

        action_dict = {"agent1": (1.0, 2.0, 3.0)}
        expected_action_dict = {"agent1": (1.0, 2.0)}

        step_result = env.step(action_dict=action_dict)
        ude_env_mock.return_value.step.assert_called_once_with(action_dict=expected_action_dict)
        assert step_result == ude_env_mock.return_value.step.return_value


    def test_step_nan_or_inf(self,
                             remote_env_adapter_mock,
                             ude_env_mock,
                             deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)

        action_dict = {"agent1": (math.nan, 2.0)}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

        action_dict = {"agent1": (1.0, math.nan)}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

        action_dict = {"agent1": (math.inf, 2.0)}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

        action_dict = {"agent1": (1.0, math.inf)}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

        action_dict = {"agent1": (-math.inf, 2.0)}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)

        action_dict = {"agent1": (1.0, -math.inf)}
        with self.assertRaises(ValueError):
            _ = env.step(action_dict=action_dict)


    def test_reset(self,
                   remote_env_adapter_mock,
                   ude_env_mock,
                   deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        obs_dict = env.reset()
        ude_env_mock.return_value.reset.assert_called_once()
        assert obs_dict == ude_env_mock.return_value.reset.return_value

    def test_close(self,
                   remote_env_adapter_mock,
                   ude_env_mock,
                   deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        env.close()
        ude_env_mock.return_value.close.assert_called_once()

    def test_observation_space(self,
                               remote_env_adapter_mock,
                               ude_env_mock,
                               deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        assert ude_env_mock.return_value.observation_space == env.observation_space

    def test_action_space(self,
                          remote_env_adapter_mock,
                          ude_env_mock,
                          deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        assert ude_env_mock.return_value.action_space == env.action_space

    def test_side_channel(self,
                          remote_env_adapter_mock,
                          ude_env_mock,
                          deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        assert ude_env_mock.return_value.side_channel == env.side_channel

    def test_get_track(self,
                       remote_env_adapter_mock,
                       ude_env_mock,
                       deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        track = env.get_track()
        deepracer_config_mock.return_value.get_track.assert_called_once()
        assert track == deepracer_config_mock.return_value.get_track.return_value

    def test_apply_track(self,
                         remote_env_adapter_mock,
                         ude_env_mock,
                         deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.track_names = frozenset({"reinvent",
                                                  "red_star_open"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        track_mock = MagicMock()
        track_mock.name = "reinvent"
        env.apply_track(track=track_mock)
        deepracer_config_mock.return_value.apply_track.assert_called_once_with(track=track_mock)

    def test_apply_track_invalid_name(self,
                                      remote_env_adapter_mock,
                                      ude_env_mock,
                                      deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.track_names = frozenset({"reinvent",
                                                  "red_star_open"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        track_mock = MagicMock()
        track_mock.name = "invalid_track"
        with self.assertRaises(ValueError):
            env.apply_track(track=track_mock)

    def test_apply_track_ignore_case_and_trim(self,
                                              remote_env_adapter_mock,
                                              ude_env_mock,
                                              deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.track_names = frozenset({"reinvent",
                                                  "red_star_open"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        track_mock = MagicMock()
        track_mock.name = "   Red_Star_Open "
        env.apply_track(track=track_mock)
        deepracer_config_mock.return_value.apply_track.assert_called_once_with(track=track_mock)

    def test_get_agent(self,
                       remote_env_adapter_mock,
                       ude_env_mock,
                       deepracer_config_mock):
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent = env.get_agent()
        deepracer_config_mock.return_value.get_agents.assert_called_once()
        assert agent == deepracer_config_mock.return_value.get_agents.return_value[0]

    def test_apply_agent(self,
                         remote_env_adapter_mock,
                         ude_env_mock,
                         deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = frozenset({"deepracer_black",
                                                  "deepracer_white",
                                                  "banana_blue"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent_mock = MagicMock()
        agent_mock.shell = "deepracer_black"
        env.apply_agent(agent=agent_mock)
        deepracer_config_mock.return_value.apply_agent.assert_called_once_with(agent=agent_mock)

    def test_apply_agent_custom_shell(self,
                                      remote_env_adapter_mock,
                                      ude_env_mock,
                                      deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = frozenset({"deepracer_black",
                                                  "deepracer_white",
                                                  "banana_blue"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent_mock = MagicMock()
        agent_mock.shell = "banana_blue"
        env.apply_agent(agent=agent_mock)
        deepracer_config_mock.return_value.apply_agent.assert_called_once_with(agent=agent_mock)

    def test_apply_agent_invalid_deepracer_shell(self,
                                                 remote_env_adapter_mock,
                                                 ude_env_mock,
                                                 deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = frozenset({"deepracer_black",
                                                  "deepracer_white",
                                                  "banana_blue"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent_mock = MagicMock()
        agent_mock.shell = "deepracer"
        with self.assertRaises(ValueError):
            env.apply_agent(agent=agent_mock)

    def test_apply_agent_invalid_color(self,
                                       remote_env_adapter_mock,
                                       ude_env_mock,
                                       deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = frozenset({"deepracer_black",
                                                  "deepracer_white",
                                                  "banana_blue"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent_mock = MagicMock()
        agent_mock.shell = "deepracer_green"
        with self.assertRaises(ValueError):
            env.apply_agent(agent=agent_mock)

    def test_apply_agent_invalid_custom_shell(self,
                                              remote_env_adapter_mock,
                                              ude_env_mock,
                                              deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = frozenset({"deepracer_black",
                                                  "deepracer_white",
                                                  "banana_blue"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent_mock = MagicMock()
        agent_mock.shell = "non_existing_custom_shell"
        with self.assertRaises(ValueError):
            env.apply_agent(agent=agent_mock)

    def test_apply_agent_ignore_case_and_trim(self,
                                              remote_env_adapter_mock,
                                              ude_env_mock,
                                              deepracer_config_mock):
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = frozenset({"deepracer_black",
                                                  "deepracer_white",
                                                  "banana_blue"})
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        agent_mock = MagicMock()
        agent_mock.shell = "   DeepRacer_White  "
        env.apply_agent(agent=agent_mock)
        assert agent_mock.shell == "deepracer_white"
        deepracer_config_mock.return_value.apply_agent.assert_called_once_with(agent=agent_mock)

    def test_track_names(self,
                         remote_env_adapter_mock,
                         ude_env_mock,
                         deepracer_config_mock):
        expected_track_names = frozenset({"reinvent",
                                          "red_star_open"})
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.track_names = expected_track_names
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        assert env.track_names == expected_track_names

    def test_shell_names(self,
                         remote_env_adapter_mock,
                         ude_env_mock,
                         deepracer_config_mock):
        expected_shell_names = frozenset({"deepracer_black",
                                          "deepracer_white",
                                          "banana_blue"})
        area_config_mock = deepracer_config_mock.return_value.get_area.return_value
        area_config_mock.shell_names = expected_shell_names
        address = "test_ip"
        env = DeepRacerEnv(address=address)
        assert env.shell_names == expected_shell_names