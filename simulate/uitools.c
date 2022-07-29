// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "uitools.h"
#include <stdio.h>
#include <string.h>


//-------------------------------- Internal GLFW callbacks ------------------------------

// update state
static void uiUpdateState(GLFWwindow* wnd) {
  // extract data from user pointer
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;

  // mouse buttons
  state->left =   (glfwGetMouseButton(wnd, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  state->right =  (glfwGetMouseButton(wnd, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);
  state->middle = (glfwGetMouseButton(wnd, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);

  // keyboard modifiers
  state->control = (glfwGetKey(wnd, GLFW_KEY_LEFT_CONTROL)==GLFW_PRESS ||
                    glfwGetKey(wnd, GLFW_KEY_RIGHT_CONTROL)==GLFW_PRESS);
  state->shift =   (glfwGetKey(wnd, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(wnd, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);
  state->alt =     (glfwGetKey(wnd, GLFW_KEY_LEFT_ALT)==GLFW_PRESS ||
                    glfwGetKey(wnd, GLFW_KEY_RIGHT_ALT)==GLFW_PRESS);

  // swap left and right if Alt
  if (state->alt) {
    int tmp = state->left;
    state->left = state->right;
    state->right = tmp;
  }

  // get mouse position, scale by buffer-to-window ratio
  double x, y;
  glfwGetCursorPos(wnd, &x, &y);
  x *= ptr->buffer2window;
  y *= ptr->buffer2window;

  // invert y to match OpenGL convention
  y = state->rect[0].height - y;

  // save
  state->dx = x - state->x;
  state->dy = y - state->y;
  state->x = x;
  state->y = y;

  // find mouse rectangle
  state->mouserect = mjr_findRect(mju_round(x), mju_round(y), state->nrect-1, state->rect+1) + 1;
}



// keyboard
static void uiKeyboard(GLFWwindow* wnd, int key, int scancode, int act, int mods) {
  // release: nothing to do
  if (act==GLFW_RELEASE) {
    return;
  }

  // extract data from user pointer
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;

  // update state
  uiUpdateState(wnd);

  // set key info
  state->type = mjEVENT_KEY;
  state->key = key;
  state->keytime = glfwGetTime();

  // application-specific processing
  ptr->uiEvent(state);
}



// mouse button
static void uiMouseButton(GLFWwindow* wnd, int button, int act, int mods) {
  // extract data from user pointer
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;

  // update state
  uiUpdateState(wnd);

  // translate button
  if (button==GLFW_MOUSE_BUTTON_LEFT) {
    button = mjBUTTON_LEFT;
  } else if (button==GLFW_MOUSE_BUTTON_RIGHT) {
    button = mjBUTTON_RIGHT;
  } else {
    button = mjBUTTON_MIDDLE;
  }

  // swap left and right if Alt
  if (glfwGetKey(wnd, GLFW_KEY_LEFT_ALT)==GLFW_PRESS ||
      glfwGetKey(wnd, GLFW_KEY_RIGHT_ALT)==GLFW_PRESS) {
    if (button==mjBUTTON_LEFT) {
      button = mjBUTTON_RIGHT;
    } else if (button==mjBUTTON_RIGHT) {
      button = mjBUTTON_LEFT;
    }
  }

  // press
  if (act==GLFW_PRESS) {
    // detect doubleclick: 250 ms
    if (button==state->button && glfwGetTime()-state->buttontime<0.25) {
      state->doubleclick = 1;
    } else {
      state->doubleclick = 0;
    }

    // set info
    state->type = mjEVENT_PRESS;
    state->button = button;
    state->buttontime = glfwGetTime();

    // start dragging
    if (state->mouserect) {
      state->dragbutton = state->button;
      state->dragrect = state->mouserect;
    }
  }

  // release
  else {
    state->type = mjEVENT_RELEASE;
  }

  // application-specific processing
  ptr->uiEvent(state);

  // stop dragging after application processing
  if (state->type==mjEVENT_RELEASE) {
    state->dragrect = 0;
    state->dragbutton = 0;
  }
}



// mouse move
static void uiMouseMove(GLFWwindow* wnd, double xpos, double ypos) {
  // extract data from user pointer
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;

  // no buttons down: nothing to do
  if (!state->left && !state->right && !state->middle) {
    return;
  }

  // update state
  uiUpdateState(wnd);

  // set move info
  state->type = mjEVENT_MOVE;

  // application-specific processing
  ptr->uiEvent(state);
}



// scroll
static void uiScroll(GLFWwindow* wnd, double xoffset, double yoffset) {
  // extract data from user pointer
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;

  // update state
  uiUpdateState(wnd);

  // set scroll info, scale by buffer-to-window ratio
  state->type = mjEVENT_SCROLL;
  state->sx = xoffset * ptr->buffer2window;
  state->sy = yoffset * ptr->buffer2window;

  // application-specific processing
  ptr->uiEvent(state);
}



// resize
static void uiResize(GLFWwindow* wnd, int width, int height) {
  // extract data from user pointer
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;

  // set layout
  ptr->uiLayout(state);

  // update state
  uiUpdateState(wnd);

  // set resize info
  state->type = mjEVENT_RESIZE;

  // stop dragging
  state->dragbutton = 0;
  state->dragrect = 0;

  // application-specific processing (unless called with 0,0 from uiModify)
  if (width && height) {
    ptr->uiEvent(state);
  }
}


static void uiRender(GLFWwindow* wnd) {
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;
  ptr->uiRender(state);
}

static void uiDrop(GLFWwindow* wnd, int count, const char** paths) {
  uiUserPointer* ptr = (uiUserPointer*)glfwGetWindowUserPointer(wnd);
  mjuiState* state = ptr->state;
  ptr->uiDrop(state, count, paths);
}

//----------------------------------- Public API ----------------------------------------

// Compute suitable font scale.
int uiFontScale(GLFWwindow* wnd) {
  // compute framebuffer-to-window ratio
  int width_win, width_buf, height;
  glfwGetWindowSize(wnd, &width_win, &height);
  glfwGetFramebufferSize(wnd, &width_buf, &height);
  double b2w = (double)width_buf / (double)width_win;

  // compute PPI
  int width_MM, height_MM;
  glfwGetMonitorPhysicalSize(glfwGetPrimaryMonitor(), &width_MM, &height_MM);
  int width_vmode = glfwGetVideoMode(glfwGetPrimaryMonitor())->width;
  double PPI = 25.4 * b2w * (double)width_vmode / (double)width_MM;

  // estimate font scaling, guard against unrealistic PPI
  int fs;
  if (width_buf>width_win) {
    fs = mju_round(b2w * 100);
  } else if (PPI>50 && PPI<350) {
    fs = mju_round(PPI);
  } else {
    fs = 150;
  }
  fs = mju_round(fs * 0.02) * 50;
  fs = mjMIN(300, mjMAX(100, fs));

  return fs;
}



// Set internal and user-supplied UI callbacks in GLFW window.
void uiSetCallback(GLFWwindow* wnd, mjuiState* state,
                   uiEventFn uiEvent, uiLayoutFn uiLayout,
                   uiRenderFn uiUserRender, uiDropFn uiUserDrop) {
  // make container with user-supplied objects and set window pointer
  uiUserPointer* ptr = (uiUserPointer*) mju_malloc(sizeof(uiUserPointer));
  ptr->state = state;
  ptr->uiEvent = uiEvent;
  ptr->uiLayout = uiLayout;
  ptr->uiRender = uiUserRender;
  ptr->uiDrop = uiUserDrop;
  glfwSetWindowUserPointer(wnd, ptr);

  // compute framebuffer-to-window pixel ratio
  int width_win, width_buf, height;
  glfwGetWindowSize(wnd, &width_win, &height);
  glfwGetFramebufferSize(wnd, &width_buf, &height);
  ptr->buffer2window = (double)width_buf / (double)width_win;

  // set internal callbacks
  glfwSetKeyCallback(wnd, uiKeyboard);
  glfwSetCursorPosCallback(wnd, uiMouseMove);
  glfwSetMouseButtonCallback(wnd, uiMouseButton);
  glfwSetScrollCallback(wnd, uiScroll);
  glfwSetWindowSizeCallback(wnd, uiResize);
  glfwSetWindowRefreshCallback(wnd, uiRender);
  glfwSetDropCallback(wnd, uiDrop);
}



// Clear UI callbacks in GLFW window.
void uiClearCallback(GLFWwindow* wnd) {
  // clear container
  if (glfwGetWindowUserPointer(wnd)) {
    mju_free(glfwGetWindowUserPointer(wnd));
    glfwSetWindowUserPointer(wnd, NULL);
  }

  // clear internal callbacks
  glfwSetKeyCallback(wnd, NULL);
  glfwSetCursorPosCallback(wnd, NULL);
  glfwSetMouseButtonCallback(wnd, NULL);
  glfwSetScrollCallback(wnd, NULL);
  glfwSetWindowSizeCallback(wnd, NULL);
  glfwSetWindowRefreshCallback(wnd, NULL);
  glfwSetDropCallback(wnd, NULL);
}



// Modify UI structure.
void uiModify(GLFWwindow* wnd, mjUI* ui, mjuiState* state, mjrContext* con) {
  mjui_resize(ui, con);
  mjr_addAux(ui->auxid, ui->width, ui->maxheight, ui->spacing.samples, con);
  uiResize(wnd, 0, 0);
  mjui_update(-1, -1, ui, state, con);
}
